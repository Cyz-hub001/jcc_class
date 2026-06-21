# 金铲铲棋子识别

自动识别《金铲铲之战》（Teamfight Tactics）游戏截图中的棋子（英雄 + 星级），采用 **YOLO 目标检测 + ResNet18 多任务分类** 的两阶段深度学习方案。

## 工作原理

```
游戏截图
   ↓
YOLO11s 目标检测 → 定位每个棋子在图中的位置
   ↓
ResNet18 多任务分类 → 对裁剪后的棋子图片同时预测「英雄名」和「星级」
   ↓
终端输出结果 + 生成标注图
```

- **第一阶段**：用 YOLO11s（Ultralytics）检测截图中所有棋子区域，输出边界框
- **第二阶段**：将每个棋子区域裁剪放大后，送入自定义 ResNet18 模型，联合预测英雄身份（8 类）和星级（1/2/3 星）

## 支持的英雄

| 中文名 | 英文名 |
|--------|--------|
| 厄斐琉斯 | Aphelios |
| 费德提克 | Fiddlesticks |
| 迦娜 | Janna |
| 蕾欧娜 | Leona |
| 莫德凯撒 | Mordekaiser |
| 努努 | Nunu |
| 辛德拉 | Syndra |
| 厄加特 | Urgot |

每个英雄支持 1★ / 2★ / 3★ 三种星级识别。

## 前置要求

- Windows 10/11
- [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU（可选，无 GPU 可降级 CPU 运行）

## 快速开始（推理）

### 1. 克隆项目

```bash
git clone https://github.com/Cyz-hub001/jcc_class.git
cd jcc_class
```

### 2. 创建 Conda 环境

**方式一：自动脚本（推荐）**

```powershell
# PowerShell
.\scripts\setup_yolo_env.ps1

# 或 CMD
scripts\setup_yolo_env.bat
```

脚本会自动创建 `jcc-yolo` conda 环境，优先安装 GPU 版 PyTorch，失败则自动降级为 CPU 版。

**方式二：手动创建**

```bash
# 有 NVIDIA GPU
conda env create -f environment-yolo.yml

# 无 GPU
conda env create -f environment-yolo-cpu.yml
```

### 3. 运行推理

```powershell
# 直接激活环境后运行
conda activate jcc-yolo
python inference.py 你的截图.png -o result.png

# 或使用 conda run（不切换终端环境）
conda run -n jcc-yolo python inference.py 你的截图.png -o result.png
```

也可以使用封装脚本：

```powershell
.\scripts\inference.ps1 你的截图.png -o result.png
```

### 4. 输出示例

程序会在终端打印识别结果，并生成标注后的图片（`-o` 指定的路径）：

```
识别到 5 个棋子:
  [1] Janna 2★   检测=95% 英雄=98% 星级=92%  框=[120, 200, 180, 280]
  [2] Aphelios 1★ 检测=91% 英雄=95% 星级=88%  框=[300, 200, 360, 280]
  [3] Mordekaiser 3★ 检测=93% 英雄=97% 星级=96%  框=[480, 200, 540, 280]
  ...
```

## 项目结构

```
jcc_class/
├── config.py              # 共享配置（英雄列表、超参数、路径）
├── inference.py           # 端到端推理：截图 → 检测 → 分类 → 标注输出
├── prepare_data.py        # 数据预处理：切分数据集 + 裁剪标注框（训练用）
├── train_classifier.py    # 训练 ResNet18 分类器
├── train_yolo.py          # 训练 YOLO11s 检测器
├── batch_test.py          # 批量推理测试脚本
├── run_all.py             # 一键全流程（训练用）
├── models/
│   └── classifier.py      # ResNet18 多任务分类模型（英雄+星级联合预测）
├── utils/
│   └── conda_yolo.py      # conda 环境管理工具函数
├── scripts/
│   ├── setup_yolo_env.ps1 # 一键创建 conda 环境
│   ├── setup_yolo_env.bat
│   ├── inference.ps1      # 推理启动脚本
│   └── inference.ps1
├── dataset/               # 处理后的数据集（YOLO格式 + 分类器裁剪图）
├── runs/
│   ├── classifier/best.pt # 分类器权重（约43MB）
│   └── detect/.../best.pt # YOLO 检测权重（约5.2MB）
├── environment-yolo.yml   # Conda 环境配置（GPU）
└── environment-yolo-cpu.yml # Conda 环境配置（CPU）
```

## 配置参数

所有可调参数集中在 `config.py`，可按需修改：

**检测参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DETECT_CONF` | 0.25 | 检测置信度阈值，越高越严格，减少误检 |
| `DETECT_IOU` | 0.45 | 非极大值抑制（NMS）IoU 阈值 |
| `YOLO_MODEL` | `yolo11s.pt` | YOLO 预训练权重文件 |

**分类参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CLS_INPUT_SIZE` | 224 | ResNet18 输入图片尺寸 |
| `CLS_EPOCHS` | 50 | 分类器训练轮数 |
| `STAR_CLASS_WEIGHTS` | `[1.0, 1.5, 5.0]` | 星级损失权重（三星样本极少，大幅加权） |
| `HERO_CLASS_WEIGHTS` | `[2.8, 2.3, ...]` | 英雄类别权重（按样本量反比自动计算） |

**通用参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CROP_PADDING` | 0.08 | 检测框裁剪时的边缘扩展比例，越大包含更多背景 |
| `SPLIT_RATIO` | `(0.8, 0.1, 0.1)` | 训练/验证/测试集划分比例 |

## 训练（进阶）

如需重新训练或微调模型：

```powershell
# 完整训练流程（数据预处理 + YOLO训练 + 分类器训练）
conda activate jcc-yolo
python run_all.py

# 单独训练 YOLO 检测器
python train_yolo.py

# 单独训练分类器（需先运行 prepare_data.py 生成裁剪数据集）
python prepare_data.py
python train_classifier.py
```

训练完成后权重自动保存至 `runs/` 对应目录，推理时会自动加载最新权重。

## 常见问题

**Q: 没有 GPU 能跑吗？**

A: 可以。`setup_yolo_env.ps1` 会自动检测并降级为 CPU 环境，推理速度稍慢但完全可用。

**Q: CUDA 版本不匹配怎么办？**

A: 编辑 `environment-yolo.yml`，将 `pytorch-cuda=12.1` 改为你系统对应的 CUDA 版本（如 `11.8`）。

**Q: `conda run` 报 UnicodeEncodeError 怎么办？**

A: 在命令前加 `chcp 65001; ` 切换终端为 UTF-8 编码，或直接 `conda activate jcc-yolo` 后再运行。

**Q: 如何添加新英雄？**

A: 1) 在 `config.py` 的 `HEROES` 列表中添加英雄英文名；2) 在标注数据集中补充对应图片；3) 重新运行 `prepare_data.py` + `train_classifier.py` 重新训练分类器。

**Q: 检测结果有遗漏或误检怎么办？**

A: 调整 `DETECT_CONF`：降低（如 0.2）可减少遗漏，升高（如 0.4）可减少误检。

## License

本项目仅供学习交流使用。
# 金铲铲棋子识别

自动识别金铲铲之战游戏截图中的棋子（英雄 + 星级），采用 **YOLO 目标检测 + ResNet18 分类** 的两阶段方案。

## 支持的英雄

| 英雄 | 英文名 |
|------|--------|
| 永恩 | Yone |
| 德莱文 | Draven |
| 奥恩 | Ornn |
| 锤石 | Thresh |
| 李青 | Lee Sin |
| 阿兹尔 | Azir |
| 塞特 | Sett |
| 斯维因 | Swain |

每个英雄支持 1/2/3 星识别。

## 快速开始（推理）

### 1. 克隆项目

```bash
git clone https://github.com/Cyz-hub001/jcc_class.git
cd jcc_class
```

### 2. 创建环境

**方式一：使用启动脚本（推荐）**

```powershell
# PowerShell
.\scripts\setup_yolo_env.ps1

# 或 CMD
scripts\setup_yolo_env.bat
```

脚本会自动创建 `jcc-yolo` conda 环境，优先 GPU 版，失败则降级 CPU 版。

**方式二：手动创建**

```bash
# 有 NVIDIA GPU
conda env create -f environment-yolo.yml

# 无 GPU
conda env create -f environment-yolo-cpu.yml
```

### 3. 运行推理

```bash
# 在 jcc-yolo 环境中运行
conda run -n jcc-yolo python inference.py 你的截图.png -o result.png
```

或使用启动脚本：

```powershell
.\scripts\inference.ps1 你的截图.png -o result.png
```

### 4. 输出示例

程序会在终端打印识别结果，并生成标注后的图片：

```
识别到 5 个棋子:
  [1] Yone 2★  检测=95% 英雄=98% 星级=92%  框=[120, 200, 180, 280]
  [2] Draven 1★  检测=91% 英雄=95% 星级=88%  框=[300, 200, 360, 280]
  ...
```

## 项目结构

```
├── config.py              # 共享配置（英雄列表、超参数等）
├── inference.py           # 端到端推理：截图 → 检测 → 分类 → 标注
├── prepare_data.py        # 数据预处理：切分 + 裁剪（训练用）
├── train_classifier.py    # 训练 ResNet18 分类器（训练用）
├── train_yolo.py          # 训练 YOLO11n 检测器（训练用）
├── run_all.py             # 一键全流程（训练用）
├── models/
│   └── classifier.py      # ResNet18 多任务分类模型
├── runs/
│   ├── classifier/best.pt # 分类器权重（43MB）
│   └── detect/.../best.pt # YOLO 检测权重（5.2MB）
├── scripts/
│   ├── setup_yolo_env.*   # 一键创建环境
│   └── inference.*        # 推理启动脚本
├── environment-yolo.yml   # Conda 环境（GPU）
└── environment-yolo-cpu.yml # Conda 环境（CPU）
```

## 配置说明

可调参数集中在 [config.py](config.py)：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DETECT_CONF` | 0.25 | 检测置信度阈值，越高越严格 |
| `DETECT_IOU` | 0.45 | 检测 NMS IoU 阈值 |
| `CROP_PADDING` | 0.08 | 检测框裁剪时的扩展比例 |

## 常见问题

**Q: 没有 GPU 能跑吗？**
A: 可以。`setup_yolo_env.ps1` 会自动降级为 CPU 环境，推理速度会慢一些。

**Q: CUDA 版本不匹配怎么办？**
A: 编辑 `environment-yolo.yml`，将 `pytorch-cuda=12.1` 改为你系统对应的版本。

**Q: 如何添加新英雄？**
A: 1) 在 `config.py` 的 `HEROES` 列表中添加英雄名；2) 重新标注数据并训练。

## License

本项目仅供学习交流使用。
