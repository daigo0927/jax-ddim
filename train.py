import argparse
from pathlib import Path
from typing import Tuple, Any
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint
import optax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from model import DiffusionModel


def create_output_dir(output_dir: Path) -> Tuple[Path, Path, Path]:
    ckpt_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        ckpt_dir.mkdir()
        log_dir.mkdir()

    return (output_dir, ckpt_dir, log_dir)


def preprocess_image(data, image_size):
    image = data['image']
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(image,
                                          (height - crop_size) // 2,
                                          (width - crop_size) // 2,
                                          crop_size,
                                          crop_size)
    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=(image_size, image_size),
                            antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_datasets(image_size: int = 64,
                     batch_size: int = 64):
    dataset_name = 'oxford_flowers102'
    split_train = 'train[:80%]+validation[:80%]+test[:80%]'
    split_val = 'train[80%:]+validation[80%:]+test[80%:]'

    preprocess_fn = partial(preprocess_image, image_size=image_size)
    
    ds_train = tfds.load(dataset_name, split=split_train, shuffle_files=True)\
                   .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)\
                   .cache()\
                   .shuffle(buffer_size=10*batch_size)\
                   .batch(batch_size, drop_remainder=True)\
                   .prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_train = tfds.as_numpy(ds_train)
                   
    ds_val = tfds.load(dataset_name, split=split_val, shuffle_files=True)\
                 .map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)\
                 .cache()\
                 .batch(batch_size, drop_remainder=True)\
                 .prefetch(buffer_size=tf.data.AUTOTUNE)
    ds_val = tfds.as_numpy(ds_val)

    return ds_train, ds_val


class TrainState(train_state.TrainState):
    batch_stats: Any


def l1_loss(predictions, targets):
    return jnp.abs(predictions - targets)


def kernel_inception_distance():
    raise NotImplementedError()


def update_ema(p_cur, p_new, momentum: float = 0.999):
    return momentum*p_cur + (1-momentum)*p_new


@jax.jit
def train_step(state, batch, rng):
    def loss_fn(params):
        outputs, mutated_vars = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch, rng, train=True,
            mutable=['batch_stats']
        )
        noises, images, pred_noises, pred_images = outputs
        
        noise_loss = l1_loss(pred_noises, noises).mean()
        image_loss = l1_loss(pred_images, images).mean()
        loss = noise_loss + image_loss
        return loss, mutated_vars
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mutated_vars), grads = grad_fn(state.params)
    state = state.apply_gradients(
        grads=grads,
        batch_stats=mutated_vars['batch_stats'])
    return state, loss
        

@partial(jax.jit, static_argnums=4)
def evaluate(state, params, rng, batch, diffusion_steps: int):
    variables = {'params': params, 'batch_stats': state.batch_stats}
    generated_images = state.apply_fn(variables,
                                      rng, batch.shape, diffusion_steps,
                                      method=DiffusionModel.generate)
    return generated_images


def run(epochs: int,
        image_size: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        val_diffusion_steps: int,
        output_dir: Path):
    tf.config.experimental.set_visible_devices([], 'GPU')
    
    output_dir, ckpt_dir, log_dir = create_output_dir(output_dir)
    summary_writer = tf.summary.create_file_writer(str(log_dir))
    
    rng = jax.random.PRNGKey(0)
    rng, key_init, key_diffusion = jax.random.split(rng, 3)

    ds_train, _ = prepare_datasets(image_size, batch_size)

    image_shape = (batch_size, image_size, image_size, 3)
    dummy = jnp.ones(image_shape, dtype=jnp.float32)

    model = DiffusionModel()
    variables = model.init(key_init, dummy, key_diffusion,
                           train=True)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=optax.adamw(learning_rate, weight_decay=weight_decay)
    )
    ema_params = state.params.copy(add_or_replace={})
    rng, rng_train, rng_val = jax.random.split(rng, 3)

    for epoch in range(epochs):
        losses = []
        pbar = tqdm(ds_train, desc=f'Epoch {epoch}')
        for images in pbar:
            rng_train, key = jax.random.split(rng_train)
            state, loss = train_step(state, images, key)

            pbar.set_postfix({'loss': f'{loss:.5f}'})
            losses.append(loss)
            ema_params = jax.tree_map(update_ema, ema_params, state.params)

        generated_images = evaluate(state,
                                    params=ema_params,
                                    rng=rng_val,
                                    batch=dummy,
                                    diffusion_steps=val_diffusion_steps)

        with summary_writer.as_default():
            tf.summary.scalar('loss', np.mean(losses), step=epoch)
            tf.summary.image('generated', generated_images, step=epoch,
                             max_outputs=8)
        save_checkpoint(ckpt_dir, state, step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDIM training')
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-diffusion-steps', type=int, default=80)
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    parser.add_argument('-o', '--output-dir', type=Path,
                        default=f'./outputs/{now}')

    args = parser.parse_args()

    run(**vars(args))
