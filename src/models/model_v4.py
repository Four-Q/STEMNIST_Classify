"""第四版 STEMNIST 多步卷积脉冲神经网络。

本模型针对 ``model_v2_with_lif.py`` 的两个主要问题进行调整：

* 使用 GroupNorm 代替 BatchNorm，避免运行统计量造成验证结果抖动；
* 不再把最终的 4x4 特征图直接做全局平均，而是联合使用空间发放图
  和分段时间发放率进行分类，保留字符笔画的位置与书写时序信息。

模型的脉冲特征提取器由 5 个卷积 LIF 层组成，并在 64 通道阶段使用
一个残差连接。输入形状为 ``[T, N, 1, 16, 16]``，输出为
``[N, num_classes]`` 的模拟 logits。
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class ConvSNNv4(nn.Module):
    """用于 35 类 STEMNIST 识别的时空残差卷积 SNN。

    ``forward`` 及发放率字典与现有 ``function_utils.py`` 兼容。
    ``forward_sequence`` 仍返回逐时间步分类电流，主 ``forward`` 则在
    该分支之外增加时空汇聚分类头，以提升相似字符的区分能力。
    """

    EXPECTED_PARAMETER_COUNT = 623_894

    def __init__(
        self,
        num_classes: int = 35,
        dropout: float = 0.2,
        tau: float = 10.0,
        logit_scale: float = 1.0,
        temporal_bins: int = 4,
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
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend 必须是非空字符串。")

        self.num_classes = num_classes
        self.dropout_probability = float(dropout)
        self.tau = float(tau)
        self.logit_scale = float(logit_scale)
        self.temporal_bins = temporal_bins
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
        self.norm1 = self._make_group_norm(num_groups=4, num_channels=16)
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
        self.norm2 = self._make_group_norm(num_groups=8, num_channels=32)
        self.lif2 = self._make_lif()
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
        self.norm3 = self._make_group_norm(num_groups=8, num_channels=64)
        self.lif3 = self._make_lif()

        # 64 通道残差脉冲块，增强深层特征而不改变空间尺寸。
        self.conv4 = layer.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False,
            step_mode="m",
        )
        self.norm4 = self._make_group_norm(num_groups=8, num_channels=64)
        self.lif4 = self._make_lif()
        self.pool2 = layer.MaxPool2d(
            kernel_size=2,
            stride=2,
            step_mode="m",
        )

        # [T,N,64,4,4] -> [T,N,96,4,4]
        self.conv5 = layer.Conv2d(
            in_channels=64,
            out_channels=96,
            kernel_size=3,
            padding=1,
            bias=False,
            step_mode="m",
        )
        self.norm5 = self._make_group_norm(num_groups=12, num_channels=96)
        self.lif5 = self._make_lif()

        # 逐时间步的轻量分类分支，维持 forward_sequence 的兼容接口。
        self.rate_classifier = layer.Linear(
            in_features=96,
            out_features=num_classes,
            step_mode="m",
        )

        # 时空读出：
        #   空间发放图 96*4*4 + temporal_bins 段的 96 维发放率。
        readout_features = 96 * 4 * 4 + temporal_bins * 96
        self.readout_norm = nn.LayerNorm(readout_features)
        self.readout_hidden = nn.Linear(readout_features, 256)
        self.readout_activation = nn.GELU()
        self.readout_dropout = nn.Dropout(p=self.dropout_probability)
        self.readout_classifier = nn.Linear(256, num_classes)

        self._initialize_weights()
        functional.set_step_mode(self, step_mode="m")
        functional.set_backend(
            self,
            backend=backend,
            instance=neuron.LIFNode,
        )

    @staticmethod
    def _make_group_norm(
        num_groups: int,
        num_channels: int,
    ) -> layer.GroupNorm:
        """创建不依赖 batch 运行统计量的多步 GroupNorm。"""

        return layer.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=1e-5,
            affine=True,
            step_mode="m",
        )

    def _make_lif(self) -> neuron.LIFNode:
        """创建统一配置的 LIF 神经元。"""

        return neuron.LIFNode(
            tau=self.tau,
            decay_input=False,
            surrogate_function=surrogate.ATan(),
            detach_reset=False,
            step_mode="m",
            backend="torch",
        )

    def _initialize_weights(self) -> None:
        """初始化卷积、归一化和读出层参数。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_inputs(inputs: torch.Tensor) -> None:
        """检查 STEMNIST 多步输入的维数和空间形状。"""

        if inputs.ndim != 5:
            raise ValueError("输入必须是 [T,N,C,H,W] 五维张量。")
        if inputs.shape[0] <= 0 or inputs.shape[1] <= 0:
            raise ValueError("时间维和 batch 维均不能为空。")
        if tuple(inputs.shape[2:]) != (1, 16, 16):
            raise ValueError("输入的通道和空间形状必须是 [1,16,16]。")

    def _encode(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """提取逐时间步的脉冲特征。"""

        self._validate_inputs(inputs)

        hidden1_spikes = self.lif1(self.norm1(self.conv1(inputs)))
        hidden2_spikes = self.lif2(self.norm2(self.conv2(hidden1_spikes)))
        hidden2_pooled = self.pool1(hidden2_spikes)

        hidden3_spikes = self.lif3(self.norm3(self.conv3(hidden2_pooled)))

        # 残差支路保留已有脉冲，0.5 的固定缩放避免发放率过快饱和。
        residual_current = self.norm4(self.conv4(hidden3_spikes))
        hidden4_spikes = self.lif4(residual_current + 0.5 * hidden3_spikes)
        hidden4_pooled = self.pool2(hidden4_spikes)

        hidden5_spikes = self.lif5(self.norm5(self.conv5(hidden4_pooled)))

        firing_rates = {
            "lif1": hidden1_spikes.detach().float().mean(),
            "lif2": hidden2_spikes.detach().float().mean(),
            # 保持 function_utils.py 中现有监控字段的名称。
            "output_lif": hidden5_spikes.detach().float().mean(),
        }
        return hidden5_spikes, firing_rates

    def _spatiotemporal_summary(
        self,
        feature_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """汇聚完整空间发放图与分段时间发放率。"""

        time_steps = feature_spikes.shape[0]
        if time_steps < self.temporal_bins:
            raise ValueError(
                f"输入时间步数 {time_steps} 小于 temporal_bins="
                f"{self.temporal_bins}。"
            )

        # 使用 float32 累加 240 步发放率，避免 AMP 下的统计精度损失。
        float_spikes = feature_spikes.float()
        spatial_rates = float_spikes.mean(dim=0).flatten(start_dim=1)

        temporal_chunks = torch.tensor_split(
            float_spikes,
            self.temporal_bins,
            dim=0,
        )
        temporal_rates = torch.cat(
            [
                chunk.mean(dim=(0, 3, 4))
                for chunk in temporal_chunks
            ],
            dim=1,
        )

        return torch.cat((spatial_rates, temporal_rates), dim=1)

    def forward_sequence(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回逐时间步分类电流和隐藏层平均发放率。"""

        feature_spikes, firing_rates = self._encode(inputs)
        global_features = feature_spikes.mean(dim=(-2, -1))
        output_currents = self.rate_classifier(global_features)
        return output_currents, firing_rates

    def forward(
        self,
        inputs: torch.Tensor,
        return_firing_rates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回融合空间位置和分段时间信息的分类 logits。"""

        feature_spikes, firing_rates = self._encode(inputs)

        # 轻量平均电流分支保证类别读出直接连接到每个时间步。
        global_features = feature_spikes.mean(dim=(-2, -1))
        mean_current_logits = self.rate_classifier(global_features).mean(dim=0)

        summary = self._spatiotemporal_summary(feature_spikes)
        summary = self.readout_norm(summary)
        summary = self.readout_hidden(summary)
        summary = self.readout_activation(summary)
        summary = self.readout_dropout(summary)
        summary_logits = self.readout_classifier(summary)

        logits = (summary_logits + mean_current_logits) * self.logit_scale

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
            f"backend={self.backend!r}, "
            "normalization='group_norm', "
            "readout='spatiotemporal_rate'"
        )


def build_model_v4(**kwargs: Any) -> ConvSNNv4:
    """创建第四版模型，并检查默认 35 类配置下的参数量。"""

    model = ConvSNNv4(**kwargs)
    parameter_count = model.parameter_count()

    if (
        model.num_classes == 35
        and model.temporal_bins == 4
        and parameter_count != model.EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            f"模型参数量应为 {model.EXPECTED_PARAMETER_COUNT}，"
            f"实际为 {parameter_count}。"
        )

    return model


# 与现有 Notebook 的类名保持兼容：
#     from src.models.model_v4 import ConvSNN
ConvSNN = ConvSNNv4
