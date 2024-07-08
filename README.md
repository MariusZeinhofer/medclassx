## medclassx
Medical image classification using jax.

## Developer guide

To contribute follow the description below.

## Setup Conda Environment

You can set up the `conda` environment and activate it

```bash
conda env create --file .conda_env.yaml
conda activate medclassx
```

## No Conda: Editable Install with PIP

In case you are not using conda you can install the package 
in editable mode using:

```bash
pip install -e ."[lint,test]"
```

## Set up Jax

The installation of Jax depends on the operating system and the hardware.
Consult [jax installation guide](https://github.com/google/jax?tab=readme-ov-file).