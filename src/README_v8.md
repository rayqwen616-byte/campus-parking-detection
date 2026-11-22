# 校园非机动车违停检测 v8（合并版）

> 由三个来源合并而来，取各自最强之处：
>
> | 来源 | 贡献 |
> |---|---|
> | **Claude v7** | `judge_core.py` 判定核心、`demo_v8.py`、`eval_events.py`、测试框架 |
> | **Kimi** | `web/index.html` 网页前端 |
> | **GPT** | 单图识别思路、中文路径处理、输入校验 |
> | **本项目原有** | 零填充配对思路、zones.json 规范、区域分割链路 |

---

## 一、v8 相对 v7 的改动

| 改动 | 文件 | 说明 |
|---|---|---|
| **数据配对合并** | `voc2yolo.py` | 零填充匹配 + `<filename>` 回退 + 尺寸校验 + 重复去重 + 宽高互换容错。实测有效样本 **864 → 895（+31）** |
| **中文路径修复** | `voc2yolo.py` | `cv2.imread` 对中文路径会静默失败，改用 `imdecode` |
| **网页演示** | `webapp.py` + `web/index.html` | 上传图片 → 出框/出区域/出 JSON；后端复用 `judge_core.Zone`，与视频判定同一套逻辑 |
| **测试修复** | `tests/run_all.py` | 修掉中文 Windows 下 subprocess 的 UTF-8 解码崩溃，全部测试可一键跑 |
| **测试补充** | `tests/test_webapp.py` | 网页后端的区域校验、编解码、脚点判定单测 |

---

## 二、目录结构

```
campus_bike_code_v8/
├── demo_v8.py            视频/图片判定主程序（v7 升级版）
├── judge_core.py         判定核心：跟踪、滞回、计时、骑手过滤、冷却、证据、企业微信
├── webapp.py             单图识别网页后端
├── web/index.html        网页前端
├── eval_events.py        事件级评测（P/R/延迟/每小时误报）
├── eval_zones.py         区域分割定量评测（多边形 IoU / 命中率 / F1）
├── sync_zones_model.py   把最新 best.pt 同步成 D:/models/zones.pt
├── voc2yolo.py           数据集转换（合并版配对：895 张）
├── zones.json            三级区域配置
├── gt_template.csv       事件评测标注模板
├── requirements.txt
└── tests/
    ├── run_all.py        一键跑全部测试
    ├── smoke_v8.py       demo_v8 端到端冒烟
    ├── test_voc2yolo.py  配对逻辑 6 种情况
    └── test_webapp.py    网页后端单测
```

---

## 三、快速开始

### 3.1 网页单图识别（推荐给老师演示）

```cmd
D:\yolo_env\Scripts\python.exe webapp.py
```

浏览器打开 <http://127.0.0.1:8000>，拖入一张照片即可。

指定模型和端口：

```cmd
D:\yolo_env\Scripts\python.exe webapp.py --model D:\models\ebike26.pt --port 8000
```

模型查找顺序：`models/ebike26.pt` → `../models/ebike26.pt` → `D:\models\ebike26.pt` → 本目录 `yolo26n.pt` → `yolo11n.pt`（最后这条会联网下载）。

### 3.2 视频判定

```cmd
D:\yolo_env\Scripts\python.exe demo_v8.py ^
    --source 视频.mp4 --model D:\models\ebike26.pt ^
    --zones zones.json --speed 10
```

单张图片（只标区域，不判时长）：

```cmd
D:\yolo_env\Scripts\python.exe demo_v8.py --source 图片.jpg --zones zones.json
```

树莓派：`--stride 5 --no-video`；企业微信：`set WECOM_WEBHOOK=...` 后加 `--notify`。

### 3.3 跑测试

```cmd
D:\yolo_env\Scripts\python.exe tests\run_all.py
```

预期输出：

```
[PASS] judge_core 判定层自测
[PASS] demo_v8 端到端冒烟
[PASS] voc2yolo 配对逻辑
[PASS] webapp 后端单测
[PASS] eval_zones 几何评测
[PASS] predict_zones 掩码提取
全部通过
```

---

## 四、核心设计（答辩可讲）

| 设计 | 原因 |
|---|---|
| 用脚点（框底边中点）判区域 | 摄像头斜视时框中心会偏移，轮胎落地点才代表车停在哪 |
| 多边形而非矩形区域 | 楼道口/消防通道形状不规则，矩形会误框合法车位 |
| 三条件判定（连续帧+停留+冷却） | 路过不报警、停放才报警、同位置不刷屏 |
| 区域滞回 `--margin-frac` | 边界抖动不清零计时 |
| `--gap` 秒级容错 | 遮挡/漏检不中断计时 |
| 骑手过滤 `--person-model` | 有人骑着的车不算违停 |
| 事件级评测 | 把「召回率 ≥0.90」落到事件上，而不是帧上 |
| 网页与视频共用 `Zone` | 单一真相源，避免两套判定逻辑不一致 |

---

## 四·五、区域分割模型（禁停区自动识别）

### 训练

```cmd
rmdir /s /q D:\zones_ds
D:\yolo_env\Scripts\python.exe zones2yolo_seg.py --src D:\zones_raw --out D:\zones_ds
D:\yolo_env\Scripts\python.exe train_zones.py --data D:\zones_ds\data.yaml ^
    --epochs 80 --imgsz 480 --batch 2 --device cpu --scratch --name zones_v3
```

### 同步权重

```cmd
D:\yolo_env\Scripts\python.exe sync_zones_model.py
```

复制成 `D:/models/zones.pt` 后，重启 `webapp.py`，「自动识别区域」按钮自动可用（无需加参数）。

### 定量评测（不看训练 mAP，看多边形吻合度）

```cmd
D:\yolo_env\Scripts\python.exe eval_zones.py --model D:/models/zones.pt --src D:\zones_raw --conf 0.25
```

输出每张图的 `GT/预测/命中/平均IoU/最高IoU`，以及总体的精确率、召回率、F1。
明细写入 `eval_zones.csv`。

## 五、已知限制

| 编号 | 限制 | 影响 |
|---|---|---|
| L1 | 网页只做单帧区域命中，不判违停 | 单图无法测停留时长，接口里叫 `possible_zone_conflict` |
| L2 | 区域自动分割未训练 | `/api/zones` 需要 `models/zones.pt`，当前返回 400 |
| L3 | 模型默认未随包提供 | 需自备 `ebike26.pt`，否则会回退到联网下载 COCO 权重 |
| L4 | 数据集仅 895 张、2 类 | 无三轮车，泛化性有限 |
| L5 | 未在硬件上实测 | 无 FPS/功耗数据 |

---

## 六、数据转换实测

```
[scan] 有效 895 张（按相对路径 864，按<filename>回退救回 31）；
       尺寸不符 1；重复 0；缺图 107；无目标 0
  [test] 44 张 / 66 个框
  [val] 134 张 / 207 个框
  [train] 717 张 / 1123 个框
```

- 对比：本项目原版 864 张，Claude v7 834 张，**v8 895 张**
- 剩余 107 个未配对：XML 里引用的 `微信图片_*.png` 在原始数据集中不存在，无法恢复

---

## 七、运行时环境

| 项 | 值 |
|---|---|
| Python | 3.14.4（`D:\yolo_env`） |
| ultralytics | 8.4.183 |
| PyTorch | 2.14.1+cpu |
| OpenCV | 5.0.0 |
| Pillow | ≥10（网页用） |

依赖见 `requirements.txt`。

---

## 八、2026-11-17 更新：区域分割实测

新增 13 张白天校园照片（B_001~B_013），与原有 13 张合计 **26 张**，完成标注并训练。

```
[scan] 找到 26 组图片+标注，跳过 0 组
  [val] 5 组 -> 6 个多边形
  [train] 21 组 -> 24 个多边形
```

训练：`yolo11n-seg` 从零，480px，80 轮，CPU，33 分钟。

| 指标 | 数值 |
|---|---|
| Box mAP50 | 0.083 |
| Mask mAP50 | 0.065 |
| forbidden 召回 | 0.714 |
| forbidden 精确率 | 0.097 |

**结论：不可用。** 可视化检验中，模型把左侧墙面当成禁停区，地面完全没识别出来。

### 原因

1. **样本太少**：21 张训练图 + 从零训练，模型学不到"地面"这一概念
2. **预训练权重下不下来**：GitHub 超时，ghproxy 镜像返回错误页，国内网络环境无法获取 `yolo11n-seg.pt`
3. 精确率 0.097 说明模型在大面积乱猜，而召回 0.714 靠的是"猜得够大"

### 下一步（二选一）

| 方案 | 做法 | 预计效果 |
|---|---|---|
| A. 攒数据 | 每类场景拍到 100 张以上再训 | Mask mAP50 0.4~0.6 |
| B. 上云微调 | 矩池云用预训练 `yolo11n-seg.pt`，30 轮 | 26 张也能到 0.3+ |

**强烈建议 B**：云端自带预训练权重，30 轮约 10 分钟，成本约 1 元。

弱模型已存档为 `D:\models\weak\zones_v2_26imgs.pt`，**未放入 `D:\models\zones.pt`**，
避免网页「自动识别区域」按钮输出错误结果。

---

## 九、本次代码改进

| 文件 | 改进 |
|---|---|
| `predict_zones.py` | 重写：`--min-area / --max-zones / --morph / --all-contours / --level` 五个可调参数；掩码先闭运算补洞再提轮廓；输出带 `_meta` 统计 |
| `webapp.py` | 自动发现区域模型（`models/zones.pt` → `D:/models/zones.pt` → 训练目录最新）；`/api/zones` 复用 `predict_zones.mask_to_polygons`，消除重复实现 |
| `tests/test_predict_zones.py` | 新增 7 项掩码提取单测（矩形、面积过滤、多轮廓、空掩码、闭运算、0/255 校验） |
| `tests/run_all.py` | 加入新测试，现共 5 组 |
| `zones2yolo_seg.py` / `train_zones.py` | 补入 v8 目录（原缺失） |

测试结果：

```
[PASS] judge_core 判定层自测
[PASS] demo_v8 端到端冒烟
[PASS] voc2yolo 配对逻辑
[PASS] webapp 后端单测
[PASS] eval_zones 几何评测
[PASS] predict_zones 掩码提取
[PASS] predict_zones 掩码提取
全部通过
```
