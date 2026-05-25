"""Dataset helpers for CIFAR-10 diffusion training."""

from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from distributed import barrier, is_main_process


def normalize_to_minus_one_to_one(x):
    return x * 2.0 - 1.0


def _ensure_cifar10_downloaded(data_dir: str, image_size: int):
    if not is_main_process():
        barrier()
        return

    download_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    datasets.CIFAR10(root=data_dir, train=True, download=True, transform=download_transform)
    datasets.CIFAR10(root=data_dir, train=False, download=True, transform=download_transform)
    barrier()


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    _ensure_cifar10_downloaded(data_dir, image_size) if distributed else None

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(normalize_to_minus_one_to_one),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(normalize_to_minus_one_to_one),
        ]
    )

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=not distributed, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=not distributed, transform=test_transform)

    train_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
