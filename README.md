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
