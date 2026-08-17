# Repository Guidelines

## 项目结构与模块组织

本仓库以 Jupyter Notebook 为主要实现形式。`src/data/` 包含数据流水线：`prepare_data.ipynb` 生成原始压力帧划分，`spike_encoding.ipynb` 执行脉冲编码，`prepare_spike_data.ipynb` 解压并校验编码结果。`tests/` 保存数据一致性与编码验证 Notebook，`AT_A_1.h5` 是验证样本。`data/` 存放输入压缩包及本地生成的 `pressure/`、`spike/` 数据；`log/` 记录实验过程。根目录的 `python_solve_spike_encoding.ipynb` 用于算法探索与对照。

## 环境、运行与开发命令

项目没有独立构建步骤。建议在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

在 JupyterLab 中按 `prepare_data.ipynb` → `spike_encoding.ipynb` → `prepare_spike_data.ipynb` 的顺序运行。若 Notebook 无法自动定位仓库，可设置 `STEMNIST_PROJECT_ROOT`。重建数据前检查 Notebook 中的 `OVERWRITE` 等配置，避免覆盖有效产物。

## 编码风格与命名约定

Python 代码使用 4 空格缩进，函数和变量采用 `snake_case`，类采用 `PascalCase`，常量采用 `UPPER_SNAKE_CASE`。优先使用 `pathlib.Path`、类型标注和带说明的异常；数据划分与采样必须显式固定种子。Notebook 应以 Markdown 单元说明阶段目标，并保持从上到下可重复执行。不要写入机器相关的绝对路径，也不要保留无关的大型输出。

## 测试指南

当前验证方式是 Notebook，而非 pytest。数据准备后运行 `tests/data_consistency_validate.ipynb`；修改编码算法后运行 `tests/validate_spike_encoding.ipynb`。提交前应确认所有单元无异常，且样本数量、形状、数据类型、划分互斥性及确定性检查均通过。新增验证文件沿用 `validate_<feature>.ipynb` 或 `<feature>_validate.ipynb` 命名。

## 数据与配置安全

生成的 `.npy`、临时编码目录、检查点和本地环境文件已被 `.gitignore` 排除。除仓库明确保留的压缩包外，不要提交派生数据、密钥、`.env` 或训练输出。更改数据格式时同步更新 manifest、metadata 和相关验证。

## 提交与拉取请求

现有历史使用简短的结果型摘要，如“完成脉冲编码，生成 spike.zip 文件”；尚无强制 Conventional Commits 规范。每次提交保持单一目的，标题直接说明结果。拉取请求需列出变更范围、数据或算法影响、复现步骤和验证结果；关联相关 issue，只有可视化结果发生变化时才附截图。
