"""第三版 STEMNIST 多步卷积脉冲神经网络。

相较于 ``model_v2.py`` 中的模型，本模型：

* 将BatchNorm替换成为GroupNorm；
* 将PLIF神经元替换成为LIF神经元；
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class ConvSNNv3(nn.Module):
    """用于 35 类 STEMNIST 识别的增强型卷积 SNN。

    输入形状为 ``[T, N, 1, 16, 16]``，输出 logits 形状为
    ``[N, num_classes]``。

    ``forward`` 和 firing-rate 字典保持与现有 ``function_utils.py``
    兼容。由于第二版没有最终输出 LIF，字典中的 ``output_lif`` 是
    第三个（最后一个）隐藏 LIF 的发放率兼容别名。
    """

    EXPECTED_PARAMETER_COUNT = 25_683

    def __init__(
        self,
        num_classes: int = 35,
        dropout: float = 0.2,
        tau: float = 10.0,
        logit_scale: float = 1.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes 必须大于 0。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")
        if not isinstance(tau, float) or tau <= 1.0:
            raise ValueError("tau 必须是大于 1.0 的 float。")
        if logit_scale <= 0.0:
            raise ValueError("logit_scale 必须大于 0。")

        self.num_classes = num_classes
        self.dropout_probability = float(dropout)
        self.tau = tau
        self.logit_scale = float(logit_scale)
        self.backend = backend

        # [T,N,1,16,16] -> [T,N,16,16,16]
        self.conv1 = layer.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=False,
            step_mode="m",
        )
        self.norm1 = layer.GroupNorm(
                    num_groups=4,
                    num_channels=16,
                    eps=1e-5,
                    affine=True,
                    step_mode="m",
                )
        self.lif1 = self._make_lif()

        # [T,N,16,16,16] -> [T,N,32,16,16]
        self.conv2 = layer.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False,
            step_mode="m",
        )
        self.norm2 = layer.GroupNorm(
                    num_groups=8,
                    num_channels=32,
                    eps=1e-5,
                    affine=True,
                    step_mode="m",
                )
        self.lif2 = self._make_lif()

        # [T,N,32,16,16] -> [T,N,32,8,8]
        self.pool1 = layer.MaxPool2d(
            kernel_size=2,
            stride=2,
            step_mode="m",
        )

        # [T,N,32,8,8] -> [T,N,64,8,8]
        self.conv3 = layer.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False,
            step_mode="m",
        )
        self.norm3 = layer.GroupNorm(
                    num_groups=8,
                    num_channels=64,
                    eps=1e-5,
                    affine=True,
                    step_mode="m",
                )
        self.lif3 = self._make_lif()

        # [T,N,64,8,8] -> [T,N,64,4,4]
        self.pool2 = layer.MaxPool2d(
            kernel_size=2,
            stride=2,
            step_mode="m",
        )

        self.dropout = layer.Dropout(
            p=self.dropout_probability,
            step_mode="m",
        )

        # 全局平均池化在 forward_sequence 中完成：
        # [T,N,64,4,4] -> [T,N,64]
        self.classifier = layer.Linear(
            in_features=64,
            out_features=num_classes,
            step_mode="m",
        )

        functional.set_step_mode(self, step_mode="m")
        functional.set_backend(
            self,
            backend=backend,
            instance=neuron.LIFNode,
        )

    def _make_lif(self) -> neuron.LIFNode:
        """创建 LIF 神经元。"""

        return neuron.LIFNode(
            tau=self.tau,
            decay_input=False,
            surrogate_function=surrogate.ATan(),
            detach_reset=False,
            step_mode="m",
            backend="torch",
        )

    def forward_sequence(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """处理多步输入并返回逐时间步模拟分类电流。

        Args:
            inputs: 形状为 ``[T,N,1,16,16]`` 的输入张量。

        Returns:
            ``output_currents`` 的形状为 ``[T,N,num_classes]``；
            ``firing_rates`` 保存三个隐藏 LIF 的平均发放率。
        """

        if inputs.ndim != 5:
            raise ValueError("输入必须是 [T,N,C,H,W] 五维张量。")
        if tuple(inputs.shape[2:]) != (1, 16, 16):
            raise ValueError("输入的通道和空间形状必须是 [1,16,16]。")

        hidden1_spikes = self.lif1(self.norm1(self.conv1(inputs)))

        hidden2_spikes = self.lif2(self.norm2(self.conv2(hidden1_spikes)))
        hidden2_pooled = self.pool1(hidden2_spikes)

        hidden3_spikes = self.lif3(self.norm3(self.conv3(hidden2_pooled)))
        hidden3_pooled = self.pool2(hidden3_spikes)

        # 对空间维执行全局平均池化，保留时间维和 batch 维。
        features = hidden3_pooled.mean(dim=(-2, -1))
        features = self.dropout(features)

        # 不再接最终输出 LIF。模拟电流保留了亚阈值信息，训练时的
        # CrossEntropyLoss 因而不会受到输出脉冲计数离散化的限制。
        output_currents = self.classifier(features)

        firing_rates = {
            "lif1": hidden1_spikes.detach().float().mean(),
            "lif2": hidden2_spikes.detach().float().mean(),
            # 与现有 function_utils.py 的硬编码监控字段保持兼容。
            "output_lif": hidden3_spikes.detach().float().mean(),
        }

        return output_currents, firing_rates

    def forward(
        self,
        inputs: torch.Tensor,
        return_firing_rates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回沿时间维平均的模拟分类 logits。"""

        output_currents, firing_rates = self.forward_sequence(inputs)
        logits = (
            output_currents.float().mean(dim=0)
            * self.logit_scale
        )

        if return_firing_rates:
            return logits, firing_rates
        return logits

    def parameter_count(self) -> int:
        """返回所有可训练参数的数量。"""

        return sum(parameter.numel() for parameter in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"dropout={self.dropout_probability}, "
            f"tau={self.tau}, logit_scale={self.logit_scale}, "
            f"backend={self.backend!r}, "
            "readout='mean_current'"
        )


def build_model_v3(**kwargs: Any) -> ConvSNNv3:
    """创建第三版模型，并检查默认 35 类配置下的参数量。"""

    model = ConvSNNv3(**kwargs)
    parameter_count = model.parameter_count()

    if (
        model.num_classes == 35
        and parameter_count != model.EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            f"模型参数量应为 {model.EXPECTED_PARAMETER_COUNT}，"
            f"实际为 {parameter_count}。"
        )

    return model


# 允许只修改 notebook 的导入路径：
#     from model_v3 import ConvSNN
ConvSNN = ConvSNNv3
