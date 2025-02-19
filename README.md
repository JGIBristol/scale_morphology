# scale_morphology
Shape analysis of Zebrafish scales

## Quick Start
1. Install `uv`
2. Change the right things in the config file `config.yml`
3. Run the scripts:
    1. `uv run efa`
    2. `uv run autoencoder`
    3. `uv run vae`

## Motivation
We want to track the shape change of Zebrafish scales through their life course,
for several reasons which I'll write about later.

## Environment
The environment here is managed with `uv`, a python package manager.

The only dependency that you need to install to run this project is `uv` itself.

> [!TIP]
> Download `uv` from [here](https://docs.astral.sh/uv/#installation).

## Scripts

### Configuration
To run these scripts, you'll need to set a few things in the configuration file:
1. The path to the input data - `binary_img_dir`. This should contain binary images of the segmented scales

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
