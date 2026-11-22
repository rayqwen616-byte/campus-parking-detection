# 校园违停检测 v10

v10 是 v9 的一次部署与可信度升级，原 v9 文档保留作历史说明。

## 本轮改动

- 网页上传限制 Base64 长度、图片字节数和像素总量，并校验真实图片格式及 JSON 请求。
- 非法区域参数会在进入判定前拒绝；区域名重复、坐标越界、无效停留时间/冷却值会给出错误。
- 网页与视频共用模型类别解析。默认 keep auto 会按实际权重启用常见非机动车类别；显式填写模型不存在的类别时会报错，避免静默漏检。
- 网页多个请求共用模型时增加推理锁，防止并发请求同时进入同一个模型实例。
- 静态文件路径校验改为路径相对关系检查。
- 启动脚本优先找项目内 .venv，其次使用原有 D:\yolo_env，再尝试 py -3；启动视频演示时不自动跑测试。
- 模型可通过环境变量指定：CAMPUS_BIKE_MODEL、CAMPUS_ZONES_MODEL。也会继续查找项目目录内和 D:\models 的权重。

## 快速使用

安装依赖：

    py -3 -m pip install -r requirements.txt

启动网页：

    py -3 webapp.py

浏览器访问 http://127.0.0.1:8000。网页默认只监听本机地址。

启动视频演示：

    py -3 demo_v8.py --source "samples\sample.mp4" --zones "samples\zones.json" --speed 10

类别默认使用 auto。如需手动指定类别，类别名必须与模型权重中的名称完全对应，例如：

    py -3 webapp.py --keep ebike,bicycle

指定模型路径：

    set CAMPUS_BIKE_MODEL=D:\models\ebike26.pt
    set CAMPUS_ZONES_MODEL=D:\models\zones.pt

也可以将权重放到当前项目的 models 文件夹。模型权重未包含在本源码目录内；没有项目权重时，检测入口可能回退到 Ultralytics 通用模型，类别范围和专用模型不同。

## 识别结论的边界

单张照片只表示目标位置与配置区域的相交情况，不能证明车辆已经停留到告警阈值。违停告警应使用视频时间轴，并用人工标注的独立视频评估误报、漏报和告警延迟。区域分割模型在 v9 更新记录中标为实验效果不足；正式演示优先手工配置区域。

## 测试

现有测试入口为：

    py -3 tests\run_all.py

本轮只做代码修改，未运行测试。
