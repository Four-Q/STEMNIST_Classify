"""约两万参数的轻量 STEMNIST 多步卷积脉冲神经网络。

本模型沿用 ``model_v4.py`` 的无归一化三层卷积 LIF 主干，以及联合
空间发放图和分段时间发放率的模拟读出方式。通过将卷积通道缩减为
8 -> 16 -> 32，并将读出隐藏维度缩减为 20，把默认参数量控制在
约两万，同时保持与 v4 相同的调用接口。
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class ConvSNNv4Light(nn.Module):
    """用于 35 类 STEMNIST 识别的约两万参数轻量 SNN。

    输入形状为 ``[T, N, 1, 16, 16]``，输出 logits 形状为
    ``[N, num_classes]``。构造参数、``forward``、``forward_sequence``
    和发放率字典均与 ``model_v4.ConvSNNv4`` 兼容。
    """

    EXPECTED_PARAMETER_COUNT = 19_443

    def __init__(
        self,
        num_classes: int = 35,
        dropout: float = 0.1,
        tau: float = 10.0,
        logit_scale: float = 1.0,
        temporal_bins: int = 4,
        readout_features: int = 20,
        backend: str = "torch",
    ) -> None:
        super().__init__()

        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes 必须是大于 0 的整数。")
        if not isinstance(dropout, Real) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")
        if not isinstance(tau, Real) or isinstance(tau, bool) or tau <= 1.0:
            raise ValueError("tau 必须是大于 1.0 的数值。")
        if not isinstance(logit_scale, Real) or logit_scale <= 0.0:
            raise ValueError("logit_scale 必须大于 0。")
        if not isinstance(temporal_bins, int) or temporal_bins <= 0:
            raise ValueError("temporal_bins 必须是大于 0 的整数。")
        if not isinstance(readout_features, int) or readout_features <= 0:
            raise ValueError("readout_features 必须是大于 0 的整数。")
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend 必须是非空字符串。")

        self.num_classes = num_classes
        self.dropout_probability = float(dropout)
        self.tau = float(tau)
        self.logit_scale = float(logit_scale)
        self.temporal_bins = temporal_bins
        self.readout_feature_count = readout_features
        self.backend = backend

        # [T,N,1,16,16] -> [T,N,8,16,16]
        self.conv1 = layer.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1,
            bias=True,
            step_mode="m",
        )
        self.lif1 = self._make_lif(v_threshold=0.5)

        # [T,N,8,16,16] -> [T,N,16,16,16] -> [T,N,16,8,8]
        self.conv2 = layer.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=True,
            step_mode="m",
        )
        self.lif2 = self._make_lif(v_threshold=0.75)
        self.pool1 = layer.MaxPool2d(
            kernel_size=2,
            stride=2,
            step_mode="m",
        )

        # [T,N,16,8,8] -> [T,N,32,8,8] -> [T,N,32,4,4]
        self.conv3 = layer.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=True,
            step_mode="m",
        )
        self.lif3 = self._make_lif(v_threshold=1.0)
        self.pool2 = layer.MaxPool2d(
            kernel_size=2,
            stride=2,
            step_mode="m",
        )

        # 32*4*4 空间发放图 + temporal_bins 个 32 维时间段发放率。
        summary_features = 32 * 4 * 4 + temporal_bins * 32
        self.readout_hidden = nn.Linear(summary_features, readout_features)
        self.readout_activation = nn.ReLU(inplace=True)
        self.readout_dropout = nn.Dropout(p=self.dropout_probability)
        self.readout_classifier = nn.Linear(readout_features, num_classes)

        self._initialize_weights()
        functional.set_step_mode(self, step_mode="m")
        functional.set_backend(
            self,
            backend=backend,
            instance=neuron.LIFNode,
        )

    def _make_lif(self, v_threshold: float) -> neuron.LIFNode:
        """创建与 v4 相同动力学配置的 LIF 神经元。"""

        return neuron.LIFNode(
            tau=self.tau,
            decay_input=False,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=False,
            step_mode="m",
            backend="torch",
        )

    def _initialize_weights(self) -> None:
        """沿用 v4 的卷积和小 logits 初始化策略。"""

        for convolution in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(
                convolution.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            nn.init.zeros_(convolution.bias)

        nn.init.kaiming_uniform_(
            self.readout_hidden.weight,
            nonlinearity="relu",
        )
        nn.init.zeros_(self.readout_hidden.bias)

        nn.init.normal_(self.readout_classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.readout_classifier.bias)

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        """检查多步输入的维数、空间形状和时间长度。"""

        if inputs.ndim != 5:
            raise ValueError("输入必须是 [T,N,C,H,W] 五维张量。")
        if inputs.shape[0] < self.temporal_bins:
            raise ValueError(
                f"输入时间步数 {inputs.shape[0]} 小于 temporal_bins="
                f"{self.temporal_bins}。"
            )
        if inputs.shape[1] <= 0:
            raise ValueError("batch 维不能为空。")
        if tuple(inputs.shape[2:]) != (1, 16, 16):
            raise ValueError("输入的通道和空间形状必须是 [1,16,16]。")

    def _encode(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """提取形状为 [T,N,32,4,4] 的脉冲特征。"""

        self._validate_inputs(inputs)

        hidden1_spikes = self.lif1(self.conv1(inputs))
        hidden2_spikes = self.lif2(self.conv2(hidden1_spikes))
        hidden2_pooled = self.pool1(hidden2_spikes)
        hidden3_spikes = self.lif3(self.conv3(hidden2_pooled))
        feature_spikes = self.pool2(hidden3_spikes)

        firing_rates = {
            "lif1": hidden1_spikes.detach().float().mean(),
            "lif2": hidden2_spikes.detach().float().mean(),
            "output_lif": hidden3_spikes.detach().float().mean(),
        }
        return feature_spikes, firing_rates

    def _spatiotemporal_summary(
        self,
        feature_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """汇聚完整空间发放图和分段时间发放率。"""

        float_spikes = feature_spikes.float()
        spatial_rates = float_spikes.mean(dim=0).flatten(start_dim=1)

        temporal_rates = torch.cat(
            [
                chunk.mean(dim=(0, 3, 4))
                for chunk in torch.tensor_split(
                    float_spikes,
                    self.temporal_bins,
                    dim=0,
                )
            ],
            dim=1,
        )
        return torch.cat((spatial_rates, temporal_rates), dim=1)

    def _sequence_summary(self, feature_spikes: torch.Tensor) -> torch.Tensor:
        """构造与主分类头维数一致的逐时间步特征。"""

        spatial_features = feature_spikes.float().flatten(start_dim=2)
        global_features = feature_spikes.float().mean(dim=(-2, -1))
        repeated_temporal_features = global_features.repeat(
            1,
            1,
            self.temporal_bins,
        )
        return torch.cat(
            (spatial_features, repeated_temporal_features),
            dim=2,
        )

    def _classify(self, summary: torch.Tensor) -> torch.Tensor:
        """使用与 v4 相同的无归一化模拟读出。"""

        hidden = self.readout_hidden(summary)
        hidden = self.readout_activation(hidden)
        hidden = self.readout_dropout(hidden)
        return self.readout_classifier(hidden) * self.logit_scale

    def forward_sequence(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回逐时间步模拟分类电流和平均发放率。"""

        feature_spikes, firing_rates = self._encode(inputs)
        output_currents = self._classify(self._sequence_summary(feature_spikes))
        return output_currents, firing_rates

    def forward(
        self,
        inputs: torch.Tensor,
        return_firing_rates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回时空发放率分类 logits。"""

        feature_spikes, firing_rates = self._encode(inputs)
        logits = self._classify(self._spatiotemporal_summary(feature_spikes))

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
            f"temporal_bins={self.temporal_bins}, "
            f"readout_features={self.readout_feature_count}, "
            f"backend={self.backend!r}, "
            "normalization='none', "
            "readout='spatiotemporal_rate', "
            "variant='light'"
        )


def build_model_v4_light(**kwargs: Any) -> ConvSNNv4Light:
    """创建轻量 v4，并检查默认 35 类配置的参数量。"""

    model = ConvSNNv4Light(**kwargs)
    parameter_count = model.parameter_count()

    if (
        model.num_classes == 35
        and model.temporal_bins == 4
        and model.readout_feature_count == 20
        and parameter_count != model.EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            f"模型参数量应为 {model.EXPECTED_PARAMETER_COUNT}，"
            f"实际为 {parameter_count}。"
        )

    return model


# 同时提供 v4 原有名称，便于现有 Notebook 只修改模块导入路径。
ConvSNNv4 = ConvSNNv4Light
build_model_v4 = build_model_v4_light
ConvSNN = ConvSNNv4Light
