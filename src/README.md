# 校园非机动车违停检测 · 矩池云跑通清单

三个文件：`data.yaml`、`train.py`、本说明。按顺序复制粘贴即可。

---

## 1. 租机器

- 平台：矩池云 https://matpool.com （题述“锯池云”应该是笔误）
- 显卡：**RTX 4090 24G，2.20 元/小时**（够用，别租 A100）
- 镜像：**Pytorch 2.0.1**（Ubuntu 20.04 / Python 3.10 / CUDA 11.8）
- 调试阶段可先租 **A2000 12G（0.90 元/小时）**，只写代码不训练

计费规则：进入“运行中”才计费，**释放即停**；启动/保存/恢复/不可用都不计费。

---

## 2. 传数据

网盘挂在实例的 `/mnt`，5 GB 免费，释放实例后数据不丢。

```bash
# 本地：先压成一个包
# campus_bike.zip = images/{train,val,test} + labels/{train,val,test}

# 实例里
cd /mnt && unzip campus_bike.zip -d /mnt/data/
```

目录要求：

```
/mnt/data/campus_bike/
├── data.yaml
├── images/train  images/val  images/test
└── labels/train  labels/val  labels/test
```

> 图片和标签**必须同名**（`a.jpg` ↔ `a.txt`），否则训练会静默跳过。

---

## 3. 装环境（1 分钟）

```bash
pip install -U ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo checks
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 4. 开跑

```bash
cd /mnt/project/campus_bike_code
tmux new -s train          # 断网/关网页都不影响
```

**先查数据（不花 GPU 钱）**

```bash
python train.py --stage check --data /mnt/data/campus_bike/data.yaml
```

看到 `[check] 通过` 再往下。

**正式训练**

```bash
python train.py --data /mnt/data/campus_bike/data.yaml --stage all \
  --epochs 200 --imgsz 960 --batch 32 --name yolo11n_960
```

`--stage all` = 训练 → test 集验证 → 导出 onnx/ncnn，一次跑完。

**断点续训**（实例释放后重新开机）

```bash
python train.py --resume --name yolo11n_960
```

**只想看指标 / 只导出**

```bash
python train.py --stage val    --name yolo11n_960
python train.py --stage export --name yolo11n_960
```

---

## 5. 训练监控

```bash
# 终端 1：曲线（租用时把 6006 端口导出，可直接在网页看）
tensorboard --logdir /mnt/runs --bind_all --port 6006

# 终端 2：显存
watch -n 2 nvidia-smi

# 终端 3：逐 epoch 指标
tail -f /mnt/runs/yolo11n_960/results.csv
```

三个判断信号：

- `box_loss` 前 20 epoch 应快速下降
- `mAP50` 到 100 epoch 后趋平，说明差不多收敛
- 训练 loss 降、验证 loss 升 → 过拟合，早停 + 加强增强

---

## 6. 结束一定要释放（否则一直扣钱）

```bash
export $(cat /proc/1/environ | tr '\0' '\n' | grep MATCLOUD_CANCELTOKEN) && \
/public/script/matncli node cancel -url https://matpool.com/api/public/node
```

---

## 7. 钱大概花多少

| 阶段 | 机时 | 金额 |
|---|---|---|
| 环境 + 数据检查 | 2 h | 4.4 元 |
| 小样本试跑（10% 数据 30 epoch） | 2 h | 4.4 元 |
| 正式训练（4000 张 200 epoch 960） | 5 h | 11.0 元 |
| 消融实验 2～5 组 | 8～20 h | 17.6～44 元 |
| 调参 + 蒸馏剪枝 | 6～12 h | 13.2～26.4 元 |
| 导出验证 | 2 h | 4.4 元 |
| **合计（含 30% 预留）** | **35～70 h** | **77～154 元** |

先跑“精简版”（约 77 元），够写论文主线；消融不够再加。

---

## 8. 常见坑

| 现象 | 处理 |
|---|---|
| `CUDA out of memory` | `--batch 16`，或 `--imgsz 800`；脚本已开 amp + cache=ram |
| 训练没开始就退出 | 看 `[check]`，八成是标签路径或名字对不上 |
| 网盘空间不足 | 只留 `best.pt` 和 `last.pt`，删掉旧 runs |
| 实例被自动释放 | 余额不足；长训练前先充够 |
| 上传很慢 | 网盘客户端换“专线”，支持断点续传 |
| 时间不对 | `export TZ='Asia/Shanghai'` |

---

## 9. 部署到树莓派（训练完之后）

```bash
# 在云上导出（脚本已做），把 weights 里的这两个目录拷回本地
#   best.onnx
#   best_ncnn_model/

# 树莓派上
pip install onnxruntime ncnn opencv-python
python -c "import cv2; print(cv2.__version__)"
```

推理骨架（违停判定核心）：

```python
foot = ((x1 + x2) / 2, y2)            # 检测框底边中点 = 车辆落脚点
if not roi.contains(foot): continue    # 不在禁停区，跳过
if track.duration < 180:   continue    # 停留不足 180 s，跳过
emit_event(spot_id, track.id)          # 生成违停事件，去重后告警
```

三个参数在配置里调，**不用重新训练**：`T`（停留时长，默认 180 s）、`d`（位移容差，0.5 m）、`conf`（0.35）。

---

## 10. 数据集去哪找

| 数据集 | 用途 | 地址 |
|---|---|---|
| VisDrone2019-DET | 预训练，含 bicycle/tricycle | https://github.com/VisDrone/VisDrone-Dataset |
| COCO 2017 | 通用预训练（矩池云 `/dataset/coco2017` 已有） | https://cocodataset.org |
| BDD100K | 夜间/雨天鲁棒性 | https://bdd-data.berkeley.edu |
| Cityscapes | 街景预训练（矩池云 `/dataset/Cityscapes` 已有） | https://www.cityscapes-dataset.com |
| Open Images V7 | 长尾补充 | https://storage.googleapis.com/openimages/web/index.html |
| Roboflow Universe | 搜 illegal parking / e-bike / scooter | https://universe.roboflow.com |

核心数据集还是得自己拍：6～8 个点位 × 3 个时段 × 4 种天气，首期 4000 张。
