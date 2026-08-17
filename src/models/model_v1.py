# SNN模型，与STEMNIST原始文章中的接近

import torch 
from torch import nn
from spikingjelly.activation_based import (
    functional,
    layer,
    neuron,
    surrogate,
)

class ConvSNN(nn.Module):
    """用于 STEMNIST 35 类识别的卷积SNN"""

    EXPECTED_PARAMETER_COUNT = 6_379
    
    def __init__(
        self,
        num_classes=35,
        dropout=0.3,
        tau=10.0,
        logit_scale=10.0,
        backend="torch",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.tau = tau
        self.logit_scale = float(logit_scale)

        # 第一组：空间特征提取 + 脉冲化
        # [T,N,1,16,16] -> [T,N,8,13,13]
        self.conv1 = layer.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=4,
            step_mode="m",
        )
        self.lif1 = self._make_lif()

        # [T,N,8,13,13] -> [T,N,8,6,6]
        self.pool1 = layer.MaxPool2d(
            kernel_size=2,
            step_mode="m",
        )

        # 第二组：继续提取空间特征
        # padding=1，因此空间尺寸保持 6×6
        self.conv2 = layer.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
            step_mode="m",
        )
        self.lif2 = self._make_lif()

        # [T,N,16,6,6] -> [T,N,16,3,3]
        self.pool2 = layer.MaxPool2d(
            kernel_size=2,
            step_mode="m",
        )

        # [T,N,16,3,3] -> [T,N,144]
        self.flatten = layer.Flatten(step_mode="m")

        self.dropout = layer.Dropout(
            p=dropout,
            step_mode="m",
        )

        # 每个类别对应一个输出神经元
        self.classifier = layer.Linear(
            in_features=16 * 3 * 3,
            out_features=num_classes,
            step_mode="m",
        )

        self.output_lif = self._make_lif()

        # m 表示 multi-step，多时间步模式
        functional.set_step_mode(self, step_mode="m")

        # 设置 LIF 神经元的计算后端
        functional.set_backend(
            self,
            backend=backend,
            instance=neuron.LIFNode,
        )

    def _make_lif(self):
        """创建统一配置的 LIF 神经元。"""

        return neuron.LIFNode(
            tau=self.tau,
            decay_input=False,
            surrogate_function=surrogate.ATan(),
            step_mode="m",
            backend="torch",
        )

    def forward_sequence(
        self,
        inputs,
    ):
        """
        参数：
            inputs: [T,N,1,16,16]

        返回：
            output_spikes: [T,N,35]
            firing_rates: 各层平均发放率
        """

        if inputs.ndim != 5:
            raise ValueError(
                "输入必须是 [T,N,C,H,W] 五维张量。"
            )

        if tuple(inputs.shape[2:]) != (1, 16, 16):
            raise ValueError(
                "输入的通道和空间尺寸必须是 [1,16,16]。"
            )

        # 第一层卷积产生连续电流，LIF 将其转换为脉冲
        hidden1_current = self.conv1(inputs)
        hidden1_spikes = self.lif1(hidden1_current)

        # 脉冲经过池化和第二层卷积
        pooled1 = self.pool1(hidden1_spikes)
        hidden2_current = self.conv2(pooled1)
        hidden2_spikes = self.lif2(hidden2_current)

        # 整理成分类器需要的特征
        pooled2 = self.pool2(hidden2_spikes)
        features = self.flatten(pooled2)
        features = self.dropout(features)

        # 分类器输出电流，最后一个 LIF 输出类别脉冲
        output_current = self.classifier(features)
        output_spikes = self.output_lif(output_current)

        firing_rates = {
            "lif1": hidden1_spikes.detach().float().mean(),
            "lif2": hidden2_spikes.detach().float().mean(),
            "output_lif": output_spikes.detach().float().mean(),
        }

        return output_spikes, firing_rates

    def forward(
        self,
        inputs,
        return_firing_rates = False,
    ):
        # 得到每个时间步的类别脉冲
        output_spikes, firing_rates = self.forward_sequence(inputs)

        # 不直接累加 240 步，避免 logits 和梯度随 T 放大。
        # 发放率范围是 [0, 1]，乘固定尺度后作为交叉熵 logits。
        spike_rates = output_spikes.float().mean(dim=0)
        logits = spike_rates * self.logit_scale

        if return_firing_rates:
            return logits, firing_rates

        return logits

    def parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )