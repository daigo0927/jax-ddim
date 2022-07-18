# Jax DDIM
Jax/Flax implementation of Denoising Diffusion Implicit Models

DDIM implementation following the keras example of [Denoising Diffusion Implicit Models](https://keras.io/examples/generative/ddim/)

## Setup

Main dependencies

- `jax==0.3.14`
- `flax==0.5.2`
- `tensorflow==2.9.1`
- `tensorflow-datasets==4.6.0`
- `tensorboard==2.9.1`

For instance, I recommend to use [GCP Vertex Workbench](https://cloud.google.com/vertex-ai/docs/workbench/managed/introduction) (managed JupyterLab environment) with GPU accelerator.
Vertex Workbench offers GPU environment and popular deep learning libraries.

## Run experiment

Run `train.py` or `train.ipynb`. Trained model and Tensorboard logs are saved under `outputs` directory by default.

According to the [Keras example](https://keras.io/examples/generative/ddim/), it is better to train at least 50 epochs for good results.

``` bash
python train.py \
--epoch 50 \
<other arguments ...>
```

## Results

Training loss and generated images for 50 epochs:

![losses](https://github.com/daigo0927/jax-ddim/blob/main/figures/losses.png)

![images](https://github.com/daigo0927/jax-ddim/blob/main/figures/generated_images.png)

## Notes

This implementation follows the Keras example implementation. You can check the detailed tips and discussion [here](https://keras.io/examples/generative/ddim/#lessons-learned)
