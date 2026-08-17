# src/data/transform.py

import numpy as np
import torch
from torch import Tensor


class Compose:
    """按照给定顺序组合多个压力数据变换。"""

    def __init__(self, transforms):
        # 转换为元组，防止外部修改原始变换列表。
        self.transforms = tuple(transforms)

    def __call__(self, frames):
        # value 的类型可能在变换过程中发生变化：
        # 开始是 NumPy 数组，经过 PressureToTensor 后变成 Tensor。
        value = frames

        # 按顺序依次执行每个变换。
        for transform in self.transforms:
            value = transform(value)

        # 约定所有变换完成后必须返回 Tensor。
        if not isinstance(value, Tensor):
            raise TypeError("压力数据 transform 必须返回 torch.Tensor")

        return value


class PressureToTensor:
    """将原始 uint8 压力帧转换成模型可以使用的 Tensor。"""

    def __init__(self, scale_to_unit=True):
        # True 表示把 [0, 255] 缩放到 [0, 1]。
        self.scale_to_unit = scale_to_unit

    def __call__(self, frames):
        # 单个样本必须包含 240 帧，每帧大小为 16×16。
        expected_shape = (240, 16, 16)

        if tuple(frames.shape) != expected_shape:
            raise ValueError(
                f"压力帧形状应为 {expected_shape}，"
                f"实际为 {tuple(frames.shape)}"
            )

        # np.load(..., mmap_mode="r") 返回的数组可能是只读的。
        # 显式复制可避免 torch.from_numpy 产生只读数组警告。
        array = np.array(frames, copy=True)

        # 转换为 float32 Tensor。
        tensor = torch.from_numpy(array).to(torch.float32)

        # 压力值默认归一化到 [0, 1]。
        if self.scale_to_unit:
            tensor.div_(255.0)

        # 原形状为 [T, H, W]。
        # 增加通道维后变成 [T, C, H, W]。
        return tensor.unsqueeze(1)


class NormalizePressure:
    """使用训练集统计量对压力数据进行标准化。"""

    def __init__(self, mean, std):
        if std <= 0:
            raise ValueError("std 必须大于 0")

        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # mean 和 std 应从训练集计算，不能使用验证集或测试集。
        return (tensor - self.mean) / self.std


def build_pressure_transform(mean=None, std=None):
    """创建默认的压力数据处理流程。"""

    # mean 和 std 必须同时提供或同时不提供。
    if (mean is None) != (std is None):
        raise ValueError("mean 和 std 必须同时提供")

    # 第一步始终是 Tensor 转换和数值缩放。
    transforms = [PressureToTensor()]

    # 如果提供了训练集统计量，再执行标准化。
    if mean is not None and std is not None:
        transforms.append(NormalizePressure(mean, std))

    return Compose(transforms)