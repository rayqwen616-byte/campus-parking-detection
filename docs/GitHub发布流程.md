# GitHub 发布流程（本项目实操笔记）

> 记录本次从零建仓到发布 Release 的完整流程，以及踩过的坑。下次做项目可直接复用。

---

## 一、建仓前准备（本地）

```bash
cd 项目目录
git init -b main
git config core.autocrlf true     # Windows 换行符
```

### .gitignore 必须包含

```
# 密钥
.env
.env.*
!.env.example

# 模型与数据集（体积大）
models/
*.pt
*.pth
*.onnx
datasets/

# 运行产物
runs/
_output/
*.mp4
*.log

# 缓存
__pycache__/
*.py[cod]
.venv/
```

> 教训：模型文件一旦提交，即使后来删除，也永久留在 Git 历史里（需 `filter-branch` 清理）。

---

## 二、多版本管理：用 Tag，不要复制文件夹

```bash
# 每个版本一个提交
git add -A
git commit -m "v4: 多边形禁停区 + 脚点判定"
git tag -f v4 -m "v4: 多边形禁停区 + 脚点判定"

# 全部版本做完后一次性推标签
git push origin --tags
```

### 好处

| 方式 | 体积 | 可对比 | 可回退 |
|---|---|---|---|
| 复制 11 个文件夹 | 11 倍 | 手动 diff | 手动 |
| **Git Tag** | 1 份 | `Compare` 按钮 | `git checkout v4` |

### 验证标签数

```bash
git ls-remote --tags origin | find /c "refs/tags"
```

**注意**：附注标签（`-a` 或 `-m`）每个会占 **2 个引用**（`refs/tags/v1` 和 `refs/tags/v1^{}`），
所以 12 个标签会显示 24。这是正常的，不是错误。

---

## 三、建远程仓库

1. <https://github.com/new>
2. 仓库名用**英文短横线**（`campus-parking-detection`）
3. 可见性 Public
4. **三个开关全部关闭**：README / .gitignore / LICENSE

> 教训：勾了任何一个，第一次 push 会被拒绝（远程已有提交，与本地历史不相关），
> 只能 `--force` 或删库重建。

---

## 四、推送

```bash
git remote add origin https://github.com/用户名/仓库名.git
git push -u origin main
git push origin --tags
```

### 常见报错对照

| 报错 | 原因 | 解决 |
|---|---|---|
| `Repository not found` | ①用户名错 ②仓库没建 ③无权限 | 打开 URL 确认能访问 |
| `not a git repository` | 不在项目目录 | `cd /d 项目绝对路径` |
| `remote origin already exists` | 重复添加 | `git remote set-url origin 新地址` |
| `User cancelled dialog` | 凭据窗口被关闭 | 重跑 push，或手动输 Token |
| `Updates were rejected` | 远程已有内容 | 删库重建，或 `--force` |

### 用户名一定核对

GitHub 账号名可能带后缀（如 `rayqwen616-byte`），
和昵称、和之前记的都可能不一样。**以浏览器地址栏为准。**

---

## 五、凭据（Windows）

首次 push 会弹 **Git Credential Manager**：

1. 点 `Sign in with your browser`
2. 浏览器点绿色 `Authorize git-ecosystem`
3. 关浏览器，回终端自动继续

授权范围含 Gists / Repositories / Workflow，属 GCM 标准范围。

**若反复失败**，用 Token：

1. <https://github.com/settings/tokens> → Generate new token (classic)
2. 勾 `repo`
3. push 时用户名填账号名，密码位置粘贴 Token

清除错误缓存：

```cmd
cmdkey /delete:git:https://github.com
```

---

## 六、发 Release 并上传模型

1. 仓库右侧 **Releases** → **Create a new release**
2. Choose a tag → 选最新版本标签
3. Title + 描述（Markdown）
4. **Attach binaries** → 拖入模型文件
5. 勾 **Set as the latest release** → Publish

### 为什么模型走 Release 而不是仓库

| 方式 | 仓库体积 | 可单独更新 | 可下载 |
|---|---|---|---|
| 提交进 Git | 永久增长 | 否 | 是 |
| **Release 附件** | 不计入 | 是 | 是 |

---

## 七、CI 配置与踩坑（重点）

### 最小可用 workflow

```yaml
name: tests
on:
  push:
    branches: [ main ]
  pull_request:
  workflow_dispatch:

jobs:
  test:
    runs-on: windows-latest
    env:
      PYTHONUTF8: "1"
      PYTHONIOENCODING: utf-8
      PYTHONUNBUFFERED: "1"
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: '3.12'
      - run: |
          python -m pip install --upgrade pip
          python -m pip install ultralytics opencv-python numpy Pillow
      - run: python src/tests/run_all.py
```

### 踩过的坑

| 坑 | 现象 | 修法 |
|---|---|---|
| **中文输出编码** | `UnicodeEncodeError: 'charmap' codec can't encode` | 设 `PYTHONUTF8=1` + 脚本里 `sys.stdout.reconfigure(encoding="utf-8")` |
| Node 20 弃用警告 | actions 用旧运行时 | `checkout@v5` / `setup-python@v6` |
| 看不到失败细节 | 只显示 exit code 1 | 测试**逐个运行**并打印名字 |

### 编码兜底写法（任何脚本都建议加）

```python
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
```

> 这个坑只有在 CI 里才会暴露：本地 Windows 中文系统默认 GBK 正常，
> 云端英文 Windows 默认 cp1252，打印中文直接崩。

---

## 八、徽章

```markdown
[![tests](https://github.com/用户/仓库/actions/workflows/tests.yml/badge.svg)](https://github.com/用户/仓库/actions/workflows/tests.yml)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
```

- 工作流徽章要**至少跑成功一次**才变绿，之前是灰色
- shields.io 的徽章由 GitHub 代理加载，国内网络也能显示

---

## 九、发布检查清单

- [ ] `.gitignore` 已排除模型、数据集、`.env`
- [ ] README 有：项目简介、快速开始、评测数据、已知限制、License
- [ ] LICENSE 文件存在（GitHub 会自动识别）
- [ ] 所有版本都有 Tag
- [ ] CI 绿色
- [ ] 徽章正常显示
- [ ] Release 已发布，模型可下载
- [ ] 下载附件验证大小正常（不是几 KB 的错误页）

---

## 十、以后每版的流程

```bash
# 1. 改代码，跑测试
python src/tests/run_all.py

# 2. 提交
git add -A
git commit -m "feat: 改了什么"

# 3. 打标签
git tag -a v12 -m "v12: 主题"

# 4. 推送（分支和标签分开推）
git push
git push origin v12

# 5. GitHub 建 Release，附新模型

# 6. 更新 README 版本表
```
