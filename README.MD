# STEMNIST_Classify

**项目介绍**：

使用SNN网络对STEMNIST原始压力帧数据以及经过**仿真神经形态系统**输出后的脉冲编码数据进行分类

## 1. 数据准备

1. 准备STEMNIST Dataset.zip放于`./data`目录下
2. 运行`src/data/prepare_data.ipynb`，将STEMNIST Dataset.zip解压并生成训练集、验证集和测试集的原始压力帧数据于`data/pressure`目录下
3. 若不存在spike.zip文件，则运行`src/data/spike_encoding.ipynb`, 将原始压力帧数据通过仿真神经形态系统进行脉冲编码，生成spike.zip文件并放于`./data`目录下（在租用的5090服务器上运行大约需要30min）
4. 运行`src/data/prepare_spike_data.ipynb`，将spike.zip解压并生成训练集、验证集和测试集的脉冲编码数据于`data/spike`目录下

## 2. RTX 5090训练

推荐环境为Linux、Python 3.12、PyTorch 2.8和CUDA 12.8。镜像已经包含CUDA版PyTorch时，确认`torch.cuda.is_available()`为`True`后安装其余依赖：

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

运行`src/train/pressure_snn_train.ipynb`训练压力数据，运行`src/train/spike_snn_train.ipynb`训练脉冲数据。Notebook默认设置`REPRODUCIBLE = True`，用于在相同硬件与软件环境中重复训练结果：

- 固定Python、NumPy、PyTorch、CUDA和DataLoader worker的随机种子；
- 启用PyTorch确定性算法并配置cuBLAS确定性工作区；
- 关闭TF32与cuDNN自动选优，使用Torch多步LIF后端；
- 保留float16自动混合精度、数据预取和内存常驻。

每次完整复现实验前必须重启内核并从头运行全部单元。实验目录名包含随机种子与确定性标记，结果分别写入`outputs/pressure/`和`outputs/spike/`。如需优先追求吞吐量，可设置`REPRODUCIBLE = False`，此时会启用TF32、cuDNN自动选优和CuPy后端，但不再承诺逐次结果一致。

