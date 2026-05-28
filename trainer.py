"""Simple CIFAR-10 DDPM training entry point."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from diffusion import GaussianDiffusion, get_beta_schedule
from data import build_cifar10_loaders
from metrics import compute_fid
from distributed import get_local_rank, get_rank, get_world_size, is_distributed, is_main_process, setup_distributed, unwrap_model
from models.unet import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    save_real_fake_panel,
    save_sample_grid,
    set_seed,
    log_wandb_artifact,
    setup_wandb,
    wandb_image_from_grid,
)


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    output_dir: str = "./runs/cifar10"
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 2e-4
    num_workers: int = 0
    image_size: int = 32
    num_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    ch: int = 128
    ch_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_every: int = 1
    sample_every: int = 1
    num_sample_images: int = 64
    wandb_project: str = "ddpm"
    wandb_entity: str = ""
    wandb_name: str = ""
    wandb_mode: str = "online"
    fid_every: int = 0
    fid_samples: int = 1000
    num_classes: int = 10
    class_dropout_prob: float = 0.1
    guidance_scale: float = 3.0
    samples_per_class: int = 8
    distributed: bool = False
    grad_accum_steps: int = 1
    resume_from: str = ""


def build_model(cfg: TrainConfig):
    model = UNet(
        in_ch=3,
        ch=cfg.ch,
        out_ch=3,
        ch_mult=cfg.ch_mult,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        dropout=cfg.dropout,
        resolution=cfg.image_size,
        num_classes=cfg.num_classes,
        class_dropout_prob=cfg.class_dropout_prob,
    )
    return model


def make_diffusion(cfg: TrainConfig):
    betas = get_beta_schedule(
        cfg.beta_schedule,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        num_diffusion_timesteps=cfg.num_timesteps,
    )
    return GaussianDiffusion(betas=betas)


def train(cfg: TrainConfig):
    if cfg.epochs < 1:
        raise ValueError("epochs must be at least 1")

    set_seed(cfg.seed)
    distributed = setup_distributed() if cfg.distributed or is_distributed() else False
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    device = torch.device(cfg.device)
    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")

    train_loader, test_loader = build_cifar10_loaders(
        cfg.data_dir,
        cfg.batch_size,
        cfg.num_workers,
        cfg.image_size,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    model = build_model(cfg).to(device)
    diffusion = make_diffusion(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    start_epoch = 0
    global_step = 0
    if cfg.resume_from:
        checkpoint = load_checkpoint(model, optimizer, cfg.resume_from, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
    if distributed and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    wandb_run = setup_wandb(cfg) if is_main_process() else None
    if wandb_run is not None:
        wandb_run.define_metric("train/global_step")
        wandb_run.define_metric("train/*", step_metric="train/global_step")
        wandb_run.define_metric("eval/*", step_metric="train/global_step")
        wandb_run.define_metric("samples/*", step_metric="train/global_step")
        wandb_run.define_metric("panels/*", step_metric="train/global_step")
    fixed_real_batch, fixed_real_labels = next(iter(test_loader))
    fixed_real_batch = fixed_real_batch.to(device)
    fixed_real_labels = fixed_real_labels.to(device)
    eval_model = unwrap_model(model)

    output_dir = Path(cfg.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    accum_steps = max(1, cfg.grad_accum_steps)
    sample_labels = torch.arange(cfg.num_classes, device=device).repeat_interleave(cfg.samples_per_class)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        if distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        accum_index = 0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{cfg.epochs}",
            leave=True,
            dynamic_ncols=True,
            disable=not is_main_process(),
        )
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device, dtype=torch.long)
            loss = diffusion.training_losses(
                model.forward,
                images,
                t,
                labels=labels,
                cond_drop_prob=cfg.class_dropout_prob,
            ).mean()
            loss_to_backward = loss / accum_steps
            accum_index += 1
            should_sync = accum_index == accum_steps
            sync_context = nullcontext() if (not distributed or should_sync) else model.no_sync()
            with sync_context:
                loss_to_backward.backward()

            running_loss += loss.item()
            global_step += 1

            if should_sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_index = 0

            if is_main_process():
                progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/global_step": global_step,
                        "train/lr": cfg.lr,
                        "train/grad_accum_steps": accum_steps,
                        "train/epoch": epoch + (accum_index / accum_steps),
                    },
                )

        progress.close()

        if accum_index != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch {epoch + 1}/{cfg.epochs} loss={avg_loss:.4f}")

        log_payload = {"train/loss": avg_loss, "epoch": epoch + 1}
        log_payload["train/global_step"] = global_step
        log_payload["train/epoch"] = epoch + 1

        if is_main_process() and (epoch + 1) % cfg.save_every == 0:
            checkpoint_path = save_checkpoint(eval_model, optimizer, output_dir, epoch + 1, cfg, global_step=global_step)
            if wandb_run is not None:
                log_wandb_artifact(
                    wandb_run,
                    name=f"checkpoint-epoch-{epoch + 1:06d}",
                    artifact_type="checkpoint",
                    file_path=checkpoint_path,
                    metadata={"epoch": epoch + 1, "global_step": global_step},
                )
        if is_main_process() and (epoch + 1) % cfg.sample_every == 0:
            try:
                samples_dir = output_dir / "samples"
                grid = save_sample_grid(
                    eval_model,
                    diffusion,
                    device,
                    samples_dir,
                    epoch + 1,
                    cfg.image_size,
                    num_images=sample_labels.shape[0],
                    labels=sample_labels,
                    guidance_scale=cfg.guidance_scale,
                    nrow=cfg.samples_per_class,
                )
                if wandb_run is not None and grid is not None:
                    log_payload["samples/grid"] = wandb_image_from_grid(grid)
                    log_wandb_artifact(
                        wandb_run,
                        name=f"samples-epoch-{epoch + 1:06d}",
                        artifact_type="sample_grid",
                        file_path=samples_dir / f"sample_{epoch + 1:06d}.png",
                        metadata={"epoch": epoch + 1, "global_step": global_step},
                    )

                fake_images = diffusion.p_sample_loop(
                    eval_model.forward,
                    shape=(min(cfg.num_sample_images, fixed_real_batch.shape[0]), 3, cfg.image_size, cfg.image_size),
                    device=device,
                    labels=fixed_real_labels[: min(cfg.num_sample_images, fixed_real_batch.shape[0])],
                    guidance_scale=cfg.guidance_scale,
                )
                panels_dir = output_dir / "panels"
                panel = save_real_fake_panel(
                    fixed_real_batch,
                    fake_images,
                    panels_dir,
                    epoch + 1,
                    num_images=min(cfg.num_sample_images, fixed_real_batch.shape[0], fake_images.shape[0]),
                )
                if wandb_run is not None and panel is not None:
                    log_payload["panels/real_vs_fake"] = wandb_image_from_grid(panel)
                    log_wandb_artifact(
                        wandb_run,
                        name=f"real-fake-panel-epoch-{epoch + 1:06d}",
                        artifact_type="real_fake_panel",
                        file_path=panels_dir / f"real_fake_{epoch + 1:06d}.png",
                        metadata={"epoch": epoch + 1, "global_step": global_step},
                    )
            except KeyboardInterrupt:
                print(f"epoch {epoch + 1}/{cfg.epochs} interrupted during sampling (KeyboardInterrupt)")
                raise
            except Exception as e:
                print(f"epoch {epoch + 1}/{cfg.epochs} sample generation failed: {e}")
                try:
                    import traceback

                    traceback.print_exc()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        if is_main_process() and cfg.fid_every > 0 and (epoch + 1) % cfg.fid_every == 0:
            fid = compute_fid(
                model=eval_model,
                diffusion=diffusion,
                loader=test_loader,
                device=device,
                image_size=cfg.image_size,
                num_samples=cfg.fid_samples,
                guidance_scale=cfg.guidance_scale,
                num_classes=cfg.num_classes,
            )
            if fid is None:
                print(
                    f"epoch {epoch + 1}/{cfg.epochs} fid=skipped (install torch-fidelity or torchmetrics[image] to enable FID)"
                )
            else:
                print(f"epoch {epoch + 1}/{cfg.epochs} fid={fid:.4f}")
                log_payload["eval/fid"] = fid

        if wandb_run is not None:
            wandb_run.log(log_payload)

        if distributed and dist.is_initialized():
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 DDPM model")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./runs/cifar10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", default="linear")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--num-sample-images", type=int, default=64)
    parser.add_argument("--wandb-project", default="ddpm")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--fid-every", type=int, default=0)
    parser.add_argument("--fid-samples", type=int, default=1000)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--class-dropout-prob", type=float, default=0.1)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--resume-from", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        ch=args.ch,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        sample_every=args.sample_every,
        num_sample_images=args.num_sample_images,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_mode=args.wandb_mode,
        fid_every=args.fid_every,
        fid_samples=args.fid_samples,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
        guidance_scale=args.guidance_scale,
        samples_per_class=args.samples_per_class,
        distributed=args.distributed,
        grad_accum_steps=args.grad_accum_steps,
        resume_from=args.resume_from,
    )
    train(cfg)
