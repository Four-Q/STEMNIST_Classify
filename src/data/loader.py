# src/data/loader.py

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import STEMNISTDataset


class LoaderConfig:
    """保存 DataLoader 的公共配置。"""

    def __init__(
        self,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        seed=42,
        drop_last=False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if num_workers < 0:
            raise ValueError("num_workers 不能小于 0")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.drop_last = drop_last


def _seed_worker(worker_id):
    """为每个 DataLoader worker 设置独立且可重复的随机种子。"""

    # 当前不直接使用 worker_id，但保留该参数是因为
    # DataLoader 的 worker_init_fn 会自动传入它。
    del worker_id

    # torch.initial_seed() 已经包含 worker 的编号信息。
    worker_seed = torch.initial_seed() % (2**32)

    # 同步设置 NumPy 和 Python random 的随机种子。
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader(
    data_root,
    split,
    data_kind,
    transform=None,
    config=None,
    shuffle=None,
):
    """创建单个 train、val 或 test DataLoader。"""

    # 避免把 LoaderConfig 实例作为函数默认参数。
    if config is None:
        config = LoaderConfig()

    # Dataset 自己负责判断 transform 是否允许使用。
    dataset = STEMNISTDataset(
        data_root=data_root,
        split=split,
        data_kind=data_kind,
        transform=transform,
    )

    # 默认只有训练集打乱顺序。
    if shuffle is None:
        shuffle = split == "train"

    # generator 控制 RandomSampler 的随机顺序。
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    # 验证集和测试集永远不丢弃最后一个不完整 batch。
    drop_last = (
        config.drop_last
        and split == "train"
    )

    # num_workers 为 0 时不能启用 persistent_workers。
    persistent_workers = config.num_workers > 0

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def create_loaders(
    data_root,
    data_kind,
    train_transform=None,
    eval_transform=None,
    config=None,
):
    """一次创建训练、验证和测试 DataLoader。"""

    if config is None:
        config = LoaderConfig()

    # 训练集可以在以后使用带数据增强的 transform。
    train_loader = create_loader(
        data_root=data_root,
        split="train",
        data_kind=data_kind,
        transform=train_transform,
        config=config,
    )

    # 验证集和测试集应使用确定性的预处理。
    val_loader = create_loader(
        data_root=data_root,
        split="val",
        data_kind=data_kind,
        transform=eval_transform,
        config=config,
    )

    test_loader = create_loader(
        data_root=data_root,
        split="test",
        data_kind=data_kind,
        transform=eval_transform,
        config=config,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }