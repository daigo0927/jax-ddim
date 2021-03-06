from dataclasses import field
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn


def sinusoidal_embedding(x,
                         min_freq: float = 1.0,
                         max_freq: float = 1000.0,
                         embedding_dims: int = 32):
    frequencies = jnp.exp(jnp.linspace(jnp.log(min_freq),
                                       jnp.log(max_freq),
                                       embedding_dims // 2))
    # x: (batch, 1, 1, 1), angular_speeds: (embedding_dims,)
    angular_speeds = 2.0*jnp.pi*frequencies
    embeddings = jnp.concatenate([jnp.sin(angular_speeds*x),
                                  jnp.cos(angular_speeds*x)], axis=3)
    return embeddings  # (batch, 1, 1, embedding_dims)


class ResidualBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, train: bool):
        input_features = x.shape[3]
        if input_features == self.features:
            residual = x
        else:
            residual = nn.Conv(self.features, kernel_size=(3, 3))(x)

        x = nn.BatchNorm(use_running_average=not train,
                         use_bias=False, use_scale=False)(x)
        x = nn.Conv(self.features, (3, 3), 1, 1)(x)
        x = nn.swish(x)
        x = nn.Conv(self.features, (3, 3), 1, 1)(x)
        x += residual
        return x


class DownBlock(nn.Module):
    features: int
    blocks: int
    
    @nn.compact
    def __call__(self, x, train: bool) -> Tuple:
        skips = []
        for _ in range(self.blocks):
            x = ResidualBlock(self.features)(x, train=train)
            skips.append(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips


def upsample2d(x,
               scale: Union[int, Tuple[int, int]],
               method: str = 'bilinear'):
    b, h, w, c = x.shape
    
    if isinstance(scale, int):
        h_out, w_out = scale*h, scale*w
    elif len(scale) == 2:
        h_out, w_out = scale[0]*h, scale[1]*w
    else:
        raise ValueError('scale argument should be either int'
                         'or Tuple[int, int]')
    
    return jax.image.resize(x, shape=(b, h_out, w_out, c), method=method)


class UpBlock(nn.Module):
    features: int
    blocks: int

    @nn.compact
    def __call__(self, x, skips: List, train: bool):
        x = upsample2d(x, scale=2, method='bilinear')
        for _ in range(self.blocks):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(self.features)(x, train=train)
        return x


class UNet(nn.Module):
    feature_stages: List[int]
    blocks: int
    min_freq: float = 1.0
    max_freq: float = 1000.0
    embedding_dims: int = 32

    @nn.compact
    def __call__(self, noisy_images, noise_variances, train: bool):
        embeddings = sinusoidal_embedding(noise_variances,
                                          min_freq=self.min_freq,
                                          max_freq=self.max_freq,
                                          embedding_dims=self.embedding_dims)
        *_, h, w, _ = noisy_images.shape
        # (b, 1, 1, embedding_dims) -> (b, h, w, embedding_dims)
        embeddings = upsample2d(embeddings, scale=(h, w), method='nearest')

        x = nn.Conv(self.feature_stages[0], (1, 1))(noisy_images)
        x = jnp.concatenate([x, embeddings], axis=-1)

        skip_stages = []
        for features in self.feature_stages[:-1]:
            x, skips = DownBlock(features, self.blocks)(x, train=train)
            skip_stages.append(skips)

        for _ in range(self.blocks):
            x = ResidualBlock(self.feature_stages[-1])(x, train=train)

        for features in reversed(self.feature_stages[:-1]):
            skips = skip_stages.pop()
            x = UpBlock(features, self.blocks)(x, skips, train=train)

        x = nn.Conv(3, (1, 1), kernel_init=nn.initializers.zeros)(x)
        return x


class DiffusionModel(nn.Module):
    # UNet parameters
    feature_stages: List[int] = field(default_factory=lambda:
                                      [32, 64, 96, 128])
    blocks: int = 2
    min_freq: float = 1.0
    max_freq: float = 1000.0
    embedding_dims: int = 32

    # Sampling (reverse diffusion) parameters
    min_signal_rate: float = 0.02
    max_signal_rate: float = 0.95

    def setup(self):
        self.normalizer = nn.BatchNorm(use_bias=False, use_scale=False)
        self.network = UNet(feature_stages=self.feature_stages,
                            blocks=self.blocks,
                            min_freq=self.min_freq,
                            max_freq=self.max_freq,
                            embedding_dims=self.embedding_dims)

    def __call__(self, images, rng, train: bool):
        images = self.normalizer(images, use_running_average=not train)
        
        rng_noises, rng_times = jax.random.split(rng)
        noises = jax.random.normal(rng_noises, images.shape, images.dtype)
        diffusion_times = jax.random.uniform(rng_times,
                                             (images.shape[0], 1, 1, 1),
                                             images.dtype)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates*images + noise_rates*noises
        
        pred_noises, pred_images = self.denoise(noisy_images,
                                                noise_rates,
                                                signal_rates,
                                                train=train)
        return noises, images, pred_noises, pred_images

    def diffusion_schedule(self, diffusion_times):
        start_angle = jnp.arccos(self.max_signal_rate)
        end_angle = jnp.arccos(self.min_signal_rate)

        diffusion_angles = start_angle \
            + diffusion_times*(end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, train: bool):
        pred_noises = self.network(noisy_images, noise_rates**2, train=train)
        pred_images = (noisy_images - noise_rates*pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        n_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate noisy image into noise/image
            ones = jnp.ones((n_images, 1, 1, 1), dtype=initial_noise.dtype)
            diffusion_times =  ones - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images,
                                                    noise_rates,
                                                    signal_rates,
                                                    train=False)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates \
                = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images \
                + next_noise_rates * pred_noises

        return pred_images

    def denormalize(self, x):
        norm_stats = self.normalizer.variables['batch_stats']
        mean = norm_stats['mean'].reshape((1, 1, 1, -1)).astype(x.dtype)
        var = norm_stats['var'].reshape((1, 1, 1, -1)).astype(x.dtype)
        std = jnp.sqrt(var + self.normalizer.epsilon)
        return std * x + mean

    def generate(self, rng, image_shape, diffusion_steps: int):
        initial_noise = jax.random.normal(rng, image_shape)
        generated_images = self.reverse_diffusion(initial_noise,
                                                  diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return jnp.clip(generated_images, 0.0, 1.0)
