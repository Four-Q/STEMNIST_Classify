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

运行`src/train/pressure_snn_train.ipynb`训练压力数据，运行`src/train/spike_snn_train.ipynb`训练脉冲数据。Notebook默认启用以下加速项：

- float16自动混合精度与梯度缩放；
- TensorFloat-32与cuDNN卷积自动调优；
- SpikingJelly CuPy多步LIF后端；
- 8个持久DataLoader worker、预取和数据内存常驻；
- 降低进度指标同步频率，并记录samples/s和峰值显存。

实验结果分别写入`outputs/pressure/`和`outputs/spike/`。若CuPy或混合精度出现兼容性问题，可在Notebook参数区将`SNN_BACKEND`设为`"torch"`或将`AMP_ENABLED`设为`False`。

