"""Top-level CIFAR-10 DDPM training entrypoint.

This wraps the reusable training logic in `trainer.py` so you can run:

    python3 train.py

or launch it from the provided bash script.
"""

from __future__ import annotations

from trainer import TrainConfig, parse_args, train


def main():
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
        distributed=args.distributed,
        grad_accum_steps=args.grad_accum_steps,
        resume_from=args.resume_from,
    )
    train(cfg)


if __name__ == "__main__":
    main()
