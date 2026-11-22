# C 任务验证报告：加速与跟踪器可行性（2026-11-18）

> 目标：① 用 OpenVINO 给本机推理加速；② 验证 Supervision 的 ByteTrack 能否替代手写的 SimpleTracker。

## 结论速览

| 任务 | 结果 | 原因 |
|---|---|---|
| OpenVINO 加速 | ❌ 不可行 | Python 3.14 没有 wheel |
| Supervision 替代跟踪器 | ❌ 不可行 | 0.31 版已移除跟踪模块 |
| ONNX 导出 | ❌ 未完成 | 官方源下载反复中断 |
| 降分辨率提速 | ✅ 可用 | 480 比 640 快 40%，但漏检增加 |

---

## 一、OpenVINO：Python 3.14 装不上

```
pip install openvino -i https://pypi.tuna.tsinghua.edu.cn/simple
ERROR: Could not find a version that satisfies the requirement openvino (from versions: none)
```

OpenVINO 官方 wheel 目前只到 **Python 3.13**，本项目环境是 **3.14.4**。

对比：`onnxruntime 1.28.0` 已装（有 3.14 wheel），但 `onnx` 装不上，导出链路走不通。

### 可行方案

| 方案 | 做法 | 代价 |
|---|---|---|
| **A. 装 Python 3.12 独立环境** | 新建 venv 专跑 OpenVINO，主项目不动 | 约 1 小时，需重新装 ultralytics |
| B. 云端导出 | 云端有 3.12，能导出 OpenVINO IR | 本机仍需 runtime → 同样要 3.12 环境 |
| C. 暂不加速 | 接受 CPU ~5 FPS | 0 |

**只有 A 能真正落地。**

---

## 二、Supervision：能装，但没有跟踪器

### 安装（有坑）

| 源 | 结果 |
|---|---|
| 清华镜像 | ❌ 多个包名被拒（supervision / pydeprecate / defusedxml 均报 no matching） |
| 官方 PyPI | ✅ supervision 0.31.7 装成功 |

依赖注意：真正的包名是 **`pydeprecate`**（不是 `deprecate`），装错会出现：

```
ImportError: cannot import name 'void' from 'deprecate'
```

### 关键发现：0.31 版没有跟踪模块

```
ls site-packages/supervision/  → 没有 tracker 目录
dir(sv) 中含 track 的属性    → 空
```

Roboflow 把跟踪拆成了独立包 `trackers`。安装它需要拉 `av`（28 MB 视频库），本机下载多次中断，**未完成验证**。

### 可用部分

`sv.PolygonZone` 仍然存在并能正常判定。但项目里已有 `judge_core.Zone`（含 signed_dist 距离），**没有替换价值**。

### 结论

**继续用 `judge_core.Tracker`**（贪心 IoU + max_lost 容忍）。它的问题（遮挡丢 ID）目前无低成本解法。

---

## 三、降分辨率实测（唯一可立即采用的提速手段）

测试条件：`B_003.jpg`（1279×1706），`ebike26.pt`，CPU，预热 3 次后取 15 次**中位数**。

| imgsz | 中位延迟 | 均值 | 检出数 |
|---|---|---|---|
| **480** | **131.6 ms** | 158.6 ms | 1 |
| **640** | **220.5 ms** | 201.1 ms | 2 |

**结论**：降到 480 快约 **40%**，但漏掉 1 个目标（召回下降）。

### 补充基准

同一张图 PyTorch CPU 640 多次测量在 **180～340 ms** 之间波动，笔记本 CPU 睿频/后台进程影响大。**对外报数据时应写区间，不要写单次值。**

---

## 四、建议

| 优先级 | 做什么 |
|---|---|
| 1 | **暂不折腾加速**，640 分辨率 + 现有代码，先把区域识别和演示做扎实 |
| 2 | 若答辩需要"提速"数据，装 Python 3.12 venv 跑 OpenVINO（方案 A） |
| 3 | 跟踪器保持 `judge_core.Tracker`，不引入 supervision |
| 4 | 需要更高帧率时，用 `--stride 2`（隔帧处理）比降分辨率更安全 |

---

## 五、环境变更记录

本次安装（对现有环境无破坏）：

| 包 | 版本 | 来源 |
|---|---|---|
| supervision | 0.31.7 | PyPI |
| pydeprecate | 0.14.0 | PyPI |
| defusedxml / scipy / matplotlib / tqdm | 最新 | 清华源 |

未安装成功：`openvino`、`onnx`、`trackers`。
