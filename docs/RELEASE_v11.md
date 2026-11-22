# Release v11 发布说明

> 直接复制下面「发布正文」到 GitHub Release 页面即可。

---

## 发布正文（复制这段）

```markdown
# v11 · 可信度与评测版

校园非机动车违停检测系统的第 11 个迭代版本。本版修复两个影响可见结果的 bug，
并产出第一份**真实**的区域识别评测数据。

## 本版重点

### 修复 1：网页标注图不显示
前端与后端重复拼接 `data:image/jpeg;base64,` 前缀，浏览器无法解析。
后端改为只返回纯 Base64，前端自行拼接。

### 修复 2：区域多边形坐标整体错位
ultralytics 返回的 `masks.data` 分辨率不是原图（1279×1706 的图返回 480×640），
旧代码按原图尺寸归一化，导致所有多边形被压缩到左上角。

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 平均 IoU | 0.0095 | **0.6605** |
| IoU≥0.5 命中 | 0 / 30 | **34 / 40** |
| F1 | 0.000 | **0.747** |

同一个模型，只改归一化方式，指标差 70 倍。

### 其它变更
- 新增 `legal` 类别（白线车位），与 `forbidden` 形成对比学习
- 网页自动区域只输出 `forbidden`，不再把通道/合法车位误判为禁停区
- 测试扩充到 6 组，全部通过
- 新增桌面应用（pywebview 原生窗口）

## 附件说明

| 文件 | 用途 | 放置位置 |
|---|---|---|
| `ebike26.pt` | 检测模型（ebike / bicycle，mAP50 0.880） | `models/ebike26.pt` |
| `zones.pt` | 区域分割模型（forbidden / passage / legal） | `models/zones.pt` |

## 快速验证

```bash
python -m pip install -r requirements.txt
python src/webapp.py --model models/ebike26.pt --zone-model models/zones.pt
# 打开 http://127.0.0.1:8000
```

## 完整变更

见 `docs/版本演进.md` 与提交历史（`git log v10..v11`）。
```

---

## 一、在 GitHub 上传模型（手把手）

### 第 1 步：准备两个文件

| 文件 | 位置 | 大小 |
|---|---|---|
| `ebike26.pt` | `D:\models\ebike26.pt` | 5.2 MB |
| `zones.pt` | `D:\models\zones.pt` | 5.9 MB |

### 第 2 步：创建 Release

1. 打开仓库页面 → 右侧 **Releases** → **Create a new release**
2. **Choose a tag** → 选择 `v11`（已存在）
3. **Release title**：

```
v11 · 可信度与评测版（含模型权重）
```

4. **Describe this release**：粘贴上面「发布正文」那段
5. 找到页面底部 **Attach binaries by dropping them here or selecting them**
6. 把 `ebike26.pt` 和 `zones.pt` 拖进去
7. 点 **Publish release**

### 第 3 步：验证

打开 `https://github.com/你的用户名/campus-parking-detection/releases/tag/v11`，
应该能看到两个附件可下载。

---

## 二、命令行上传（可选，需 gh CLI）

如果装了 GitHub CLI：

```bash
cd /d %USERPROFILE%\Desktop\project\campus-parking-detection
gh release create v11 --title "v11 · 可信度与评测版" --notes-file docs/RELEASE_v11.md ^
   D:\models\ebike26.pt D:\models\zones.pt
```

> 注意：`gh` 需要先 `gh auth login`。

---

## 三、README 里的下载说明

发布后，可以把下面这段补进 README 的「模型权重」小节：

```markdown
| 模型 | 下载 |
|---|---|
| ebike26.pt | [Release v11](https://github.com/你的用户名/campus-parking-detection/releases/tag/v11) |
| zones.pt | [Release v11](https://github.com/你的用户名/campus-parking-detection/releases/tag/v11) |
```

---

## 四、为什么模型不进 Git

| 原因 | 说明 |
|---|---|
| 体积 | 每个 5~6 MB，长期迭代会撑爆仓库历史 |
| 可替换 | 权重可用数据重训，代码才是核心资产 |
| Git 机制 | Git 存全量快照，删了也仍占体积（需 filter-branch） |

**Release 附件是最合适的存放方式**：可下载、不打进历史、可单独更新。

---

## 五、以后每版的发布流程

```
1. 代码改完 → 跑 tests/run_all.py 全绿
2. git commit -m "feat: xxx"
3. git tag -a v12 -m "v12: 做了什么"
4. git push origin main --tags
5. GitHub 建 Release，附新权重
6. 更新 README 版本表
```
