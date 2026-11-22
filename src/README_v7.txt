校园非机动车违停检测 —— v7 完整包
==================================

一、包里有什么
  新增
    judge_core.py     判定层核心：跟踪、区域滞回、计时、骑手过滤、冷却、证据截图、企业微信推送、事件级评测函数
    demo_v7.py        主程序（替代 demo_v6.py；zones.json 格式不变）
    eval_events.py    事件级评测：系统告警 vs 人工标注的真实违停事件
    gt_template.csv   评测标注模板（只有表头）
    requirements.txt  依赖清单
    tests/            自动测试（python tests/run_all.py）
  修改
    voc2yolo.py       配不上的 XML 改按 <filename> 回退配对，并校验尺寸；输出 unpaired.txt
    pick_roi.py       兼容中文路径；多打印一行可直接粘进 zones.json 的 "points"
  未改动（原样保留；仅 plot_curves.py 的开头说明文字改成原始字符串 r"""，消除 Python 3.14 的转义警告）
    demo_v4.py demo_v5.py demo_v6.py zones2yolo_seg.py train_zones.py predict_zones.py
    rename_photos.py train.py auto_label.py plot_curves.py make_charts.py metrics.csv zones.json

二、快速开始（Windows）
  cd /d 本文件夹所在路径
  D:\yolo_env\Scripts\python.exe demo_v7.py ^
      --source 视频.mp4 --model D:\models\ebike26.pt --zones zones.json --speed 10
  单张图片（解决 B1）：--source 图片.jpg   只标出每辆车落在哪个区域，不判定时长
  树莓派：加 --stride 5 --no-video   每 5 帧处理 1 帧，不输出视频
  企业微信：先 set WECOM_WEBHOOK=你的机器人地址，再加 --notify
  排除有人骑/扶着的车：加 --person-model yolo11n.pt（COCO 模型，类别 0 = person）

三、v7 相对 v6 的改动
  判定  脚点进出区域加滞回（--margin-frac），边界抖动不再清零计时
  判定  计时单位是秒；区域内漏检/遮挡 --gap 秒内不中断（取代 --vote 帧数）
  判定  车身中心位移超过对角线的 --move 倍 = 骑行/推行，重新计时
  判定  冷却 = 同区域 + 空间距离，区域自带的 cooldown 优先（取代 80px 网格）
  判定  先按类别过滤再跟踪；重叠区域取最严重等级
  跟踪  全局按 IoU 降序配对（与检测顺序无关）；丢失的轨迹保留 --gap 秒
  输出  告警时保存带框截图到 <out>/evidence；events.csv 多一列"证据截图"；没有事件也写表头
  参数  --gap / --rider-min 的单位是"视频真实秒"，程序内部自动乘以 --speed；
        zones.json 里的 dwell / cooldown 和 v6 一样，是加速后的秒数

四、事件级评测（把"召回率 ≥ 0.90"落到证据上）
  1. 在固定机位录一段真实视频，人工标出每次真实违停：复制 gt_template.csv，每行  区域,开始(s),结束(s)
  2. 用 --speed 1 跑 demo_v7.py（时间就是视频真实秒），得到 events.csv
  3. python eval_events.py --events runs\demo7\events.csv --gt gt.csv --duration 视频总秒数
  输出 precision / recall / 平均告警延迟 / 每小时误报数。
  注意：demo.mp4 这类"几张照片拼接 + 共用一份 zones.json"的演示，不能当作评测数据。

五、数据转换（voc2yolo.py）
  python voc2yolo.py --src <VOC目录> --out <YOLO目录>
  终端会打印"按文件名 X 张，按<filename>回退救回 Y 张，仍未配上 Z 个"，未配上的清单在 <out>\unpaired.txt。
  能救回多少取决于 XML 里的 <filename> 是否真的指向原图；救不回来的需要手工核对。
  重训时请固定划分（--seed），并保证测试集只含本校图片、不参与任何调参。

六、区域配置（zones.json）
  字段：name / level(1最高) / dwell(秒) / cooldown(秒) / color(BGR) / points(0~1 归一化多边形)
  每个摄像头一份 zones.json，按该画面里真实的禁停位置重画。
  画法：python pick_roi.py 截图.jpg，点选后把打印出的 "points": [...] 粘进 zones.json。
  predict_zones.py 生成的 zones_auto.json 格式兼容，但目前分割模型还不可用（mAP=0），需人工核对。

七、测试
  python tests/run_all.py
  覆盖：判定层 12 项合成场景（边界抖动、遮挡、推行、骑手离开、冷却、重叠区域、跟踪顺序、推送限频、
  事件评测、配置报错、证据截图、真实 zones.json 格式）；demo_v7 端到端 6 项（假检测器 + 现场合成视频）；
  voc2yolo 配对 6 种情况。
  这些只证明逻辑和流程正确，不证明在真实模型、真实视频上的效果。

八、已知限制
  1. 测试用的是合成数据和假检测器，没有在 ebike26.pt + 真实视频上验证过；请先用 --speed 10 对比 v6 的结果。
  2. 阈值（10/60/180 秒、--gap 3、--move 0.35、滞回宽度）都是经验值，需要用第四节的评测数据来调。
  3. 骑手过滤靠额外的 person 模型；路人恰好走过车旁、或车主站在车旁不动，都可能误判，需实测。
  4. 摄像头模式的时间用系统时钟；推送只发文字，不带图片；3 级告警和被限频的消息只在程序结束时汇总发送。
  5. 以下问题本包没有处理：检测数据划分的近重复泄漏、区域分割从零训练（应换预训练权重）、
     实验的多种子/batch 对照、边缘设备（树莓派）实测速度。
