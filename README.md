# DDPM CIFAR-10 PyTorch Reimplementation

This repository contains a compact PyTorch port of the CIFAR-10 diffusion model from hojonathanho/diffusion.

## What is included

- `models/unet.py`: UNet backbone with residual and attention blocks
- `models/nn.py`: helper layers and timestep embeddings
- `diffusion.py`: DDPM beta schedule, training loss, and sampling utilities
- `trainer.py`: simple CIFAR-10 training entry point using torchvision

## Install

```bash
pip install -r requirements.txt
```

## Train on CIFAR-10

```bash
bash scripts/run_train_cifar.sh
```

To use multiple GPUs on one machine, set `NUM_GPUS`:

```bash
NUM_GPUS=4 bash scripts/run_train_cifar.sh
```

To resume from a saved checkpoint and use gradient accumulation:

```bash
RESUME_FROM=./runs/cifar10/checkpoint_000010.pt GRAD_ACCUM_STEPS=4 bash scripts/run_train_cifar.sh
```

When W&B logging is enabled, set `WANDB_PROJECT` and `WANDB_API_KEY` first, for example with `wandb login`.

If you want a quicker smoke run, use fewer epochs and sample images more often:

```bash
EPOCHS=1 BATCH_SIZE=8 SAVE_EVERY=1 SAMPLE_EVERY=1 bash scripts/run_train_cifar.sh
```

You can also call the Python entrypoint directly:

```bash
python3 train.py --epochs 1 --batch-size 8 --sample-every 1 --save-every 1
```
