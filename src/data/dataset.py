# src/data/dataset.py

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .transform import PressureToTensor


# 不同数据类型对应的文件名。
DATA_FILENAMES = {
    "pressure": "pressure.npy",
    "spike": "spike.npy",
}

# 两类数据的单样本形状完全相同。
EXPECTED_SAMPLE_SHAPE = (240, 16, 16)

# 当前 STEMNIST 数据集包含 A-Z 和 1-9，共 35 类。
EXPECTED_CLASS_COUNT = 35


class STEMNISTDataset(Dataset):
    """同时支持压力帧和脉冲数据的 STEMNIST Dataset。"""

    def __init__(
        self,
        data_root,
        split,
        data_kind,
        transform=None,
        in_memory=False,
    ):
        # 只接受仓库已经生成的三个数据划分。
        valid_splits = ("train", "val", "test")

        if split not in valid_splits:
            raise ValueError(
                f"未知数据划分：{split}，"
                f"可选值为 {valid_splits}"
            )

        # 使用明确的字符串区分压力帧和脉冲数据。
        valid_data_kinds = ("pressure", "spike")

        if data_kind not in valid_data_kinds:
            raise ValueError(
                f"未知数据类型：{data_kind}，"
                f"可选值为 {valid_data_kinds}"
            )

        # 通过异常强制保证脉冲数据不会使用 transform。
        # 这样可以避免误把脉冲值除以 255 或执行数据增强。
        if data_kind == "spike" and transform is not None:
            raise ValueError("脉冲数据不允许使用 transform")

        self.data_root = Path(data_root)
        self.split = split
        self.data_kind = data_kind
        self.in_memory = bool(in_memory)

        # 例如：
        # data/pressure/train/pressure.npy
        # data/spike/train/spike.npy
        split_directory = self.data_root / data_kind / split

        self.data_path = (
            split_directory / DATA_FILENAMES[data_kind]
        )
        self.manifest_path = (
            split_directory / "manifest.csv"
        )

        # 尽早检查路径，避免训练开始后才发现文件不存在。
        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"找不到数据文件：{self.data_path}"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"找不到 manifest：{self.manifest_path}"
            )

        # manifest 保存样本标签、sample_id 和数组行号。
        self.rows = self._read_manifest()

        if self.in_memory:
            # Linux 训练服务器内存充足时，可将整个划分读入内存。
            # DataLoader 使用 fork worker 时，这些只读数据页可由 worker
            # 共享，能够避免首轮训练中的随机磁盘读取。
            array = np.load(
                self.data_path,
                allow_pickle=False,
            )
        else:
            # 默认仍使用内存映射，避免普通开发环境占用过多内存。
            array = np.load(
                self.data_path,
                mmap_mode="r",
                allow_pickle=False,
            )

        # 完整数组必须是 [N, T, H, W]。
        if array.ndim != 4:
            raise ValueError(
                f"数据必须是四维数组，实际形状为 {array.shape}"
            )

        # 检查单样本的时间帧数和空间尺寸。
        actual_sample_shape = tuple(array.shape[1:])

        if actual_sample_shape != EXPECTED_SAMPLE_SHAPE:
            raise ValueError(
                f"样本形状应为 {EXPECTED_SAMPLE_SHAPE}，"
                f"实际为 {actual_sample_shape}"
            )

        # 当前压力和脉冲数据都保存为 uint8。
        if array.dtype != np.uint8:
            raise ValueError(
                f"数据类型应为 uint8，实际为 {array.dtype}"
            )

        # 数组第一维必须和 manifest 行数相同。
        if array.shape[0] != len(self.rows):
            raise ValueError(
                "数据和 manifest 长度不一致："
                f"{array.shape[0]} != {len(self.rows)}"
            )

        if self.in_memory:
            # 保留已加载数组，后续读取不会再访问磁盘。
            self._data = array
        else:
            # mmap 模式只在初始化阶段检查元数据，worker 首次读取时
            # 再各自打开文件映射。
            del array
            self._data = None

        if data_kind == "pressure":
            # 即使调用者没有传入 transform，也需要进行基础的
            # Tensor 转换、数值缩放和通道维添加。
            if transform is None:
                self.transform = PressureToTensor()
            else:
                self.transform = transform
        else:
            # 脉冲数据明确不使用 transform。
            self.transform = None

        # 根据 manifest 构建类别列表。
        self.classes = self._build_classes()

    def _read_manifest(self):
        """读取并检查当前划分的 manifest。"""

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            # pressure manifest 使用 split 字段。
            # spike manifest 使用 source_split 字段。
            if self.data_kind == "pressure":
                split_field = "split"
            else:
                split_field = "source_split"

            # 两种 manifest 都必须包含这些基础字段。
            required_fields = {
                "row_index",
                "sample_id",
                "label",
                "label_index",
                split_field,
            }

            available_fields = set(reader.fieldnames or [])
            missing_fields = required_fields - available_fields

            if missing_fields:
                raise ValueError(
                    "manifest 缺少字段："
                    f"{sorted(missing_fields)}"
                )

            rows = list(reader)

        # 检查 manifest 行号是否与数组索引严格对应。
        for expected_index, row in enumerate(rows):
            actual_index = int(row["row_index"])

            if actual_index != expected_index:
                raise ValueError(
                    f"manifest 第 {expected_index} 行的 "
                    f"row_index 为 {actual_index}"
                )

            # 防止错误地把其他划分的 manifest 放入当前目录。
            if row[split_field] != self.split:
                raise ValueError(
                    f"manifest 中出现错误划分："
                    f"{row[split_field]}"
                )

        return rows

    def _build_classes(self):
        """根据 label_index 构建有序类别列表。"""

        labels = {}

        for row in self.rows:
            label_index = int(row["label_index"])
            label = row["label"]

            # 同一个索引不能对应两个不同标签。
            if (
                label_index in labels
                and labels[label_index] != label
            ):
                raise ValueError(
                    f"标签索引 {label_index} 映射不一致"
                )

            labels[label_index] = label

        # 当前数据集的标签索引应连续覆盖 0-34。
        expected_indices = list(
            range(EXPECTED_CLASS_COUNT)
        )

        if sorted(labels) != expected_indices:
            raise ValueError(
                "manifest 未包含完整的 35 个类别"
            )

        # 最终顺序为：
        # A-Z 对应 0-25，1-9 对应 26-34。
        return [
            labels[index]
            for index in expected_indices
        ]

    def _open_array(self):
        """返回内存数组，或在首次读取时打开文件映射。"""

        if self._data is None:
            self._data = np.load(
                self.data_path,
                mmap_mode="r",
                allow_pickle=False,
            )

        return self._data

    def __getstate__(self):
        """确保 Windows worker 不直接继承已打开的 memmap。"""

        # mmap 句柄不应被 spawn worker 直接继承；内存常驻模式则保留
        # 数组。Linux 默认使用 fork，内存页不会因为只读访问而复制。
        state = self.__dict__.copy()
        if not self.in_memory:
            state["_data"] = None

        return state

    def __len__(self):
        """返回当前划分中的样本数量。"""

        return len(self.rows)

    def __getitem__(self, index):
        """读取一个样本并返回 inputs 和 target。"""

        # 从内存映射数组中只取当前样本。
        frames = self._open_array()[index]

        if self.data_kind == "pressure":
            # 压力数据使用调用者提供的 transform，
            # 或初始化时创建的默认 PressureToTensor。
            inputs = self.transform(frames)
        else:
            # 脉冲值本身已经是 0 或 1。
            # 此处只转换类型和增加通道维，不能除以 255。
            array = np.array(
                frames,
                dtype=np.uint8,
                copy=True,
            )

            inputs = torch.from_numpy(array)
            inputs = inputs.to(torch.float32)

            # [T, H, W] -> [T, C, H, W]
            inputs = inputs.unsqueeze(1)

        # 两种输入必须具有相同的模型接口。
        expected_shape = (240, 1, 16, 16)
        actual_shape = tuple(inputs.shape)

        if actual_shape != expected_shape:
            raise ValueError(
                f"模型输入形状应为 {expected_shape}，"
                f"实际为 {actual_shape}"
            )

        # CrossEntropyLoss 需要 long 类型的类别索引。
        target = torch.tensor(
            int(self.rows[index]["label_index"]),
            dtype=torch.long,
        )

        # contiguous 可以避免后续模型遇到非连续 Tensor。
        return inputs.contiguous(), target

    def sample_info(self, index):
        """返回样本的 manifest 信息，用于调试和结果分析。"""

        # 返回副本，避免调用者修改 Dataset 内部状态。
        return dict(self.rows[index])
