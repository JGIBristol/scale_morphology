# scale_morphology
Shape analysis of Zebrafish scales

## Motivation

## Environment
The environment here is managed with `uv`, a python package manager.

The only dependency that you need to install to run this project is `uv` itself.

> [!TIP]
> download `uv` from [the instructions here](https://docs.astral.sh/uv/#installation).

## Scripts
I'm envisioning three scripts...

```
uv run efa
```
Run the Elliptic Fourier Analysis on the scales.

```
uv run autoencoder
```
Train and perform embedding with a simple autoencoder.

```
uv run vae
```
Train and perform embedding with a variational autoencoder.
