# STEMNIST_Classify

**项目介绍**：

使用SNN网络对STEMNIST原始压力帧数据以及经过**仿真神经形态系统**输出后的脉冲编码数据进行分类

## 1. 数据准备

1. 准备STEMNIST Dataset.zip放于`./data`目录下
2. 运行`src/data/prepare_data.py`，将STEMNIST Dataset.zip解压并生成训练集、验证集和测试集的原始压力帧数据与`data/pressure`目录下
3. 若不存在spike.zip文件，则运行`src/data/spike_encoding.ipynb`, 将原始压力帧数据通过仿真神经形态系统进行脉冲编码，生成spike.zip文件并放于`./data`目录下（在租用的5090服务器上运行大约需要30min）
4. 运行`src/data/prepare_spike_data.py`，将spike.zip解压并生成训练集、验证集和测试集的脉冲编码数据与`data/spike`目录下

