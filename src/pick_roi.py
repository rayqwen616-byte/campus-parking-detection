# -*- coding: utf-8 -*-
"""
pick_roi.py —— 鼠标点几下，自动生成多边形 ROI 坐标

用法：
    python pick_roi.py D:\某个截图.jpg

操作：
    1. 窗口弹出后，用鼠标依次点击禁停区的各个角
    2. 至少点 3 个点
    3. 按 空格 结束（或按 Enter）
    4. 终端会打印可直接粘贴的 --roi 参数
    5. 按 ESC 可取消

注意：点完会自动缩放到 1280 宽显示，坐标会自动换算回原图比例。
"""
import sys

import cv2

POINTS = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x, y))
        img = param["img"].copy()
        for p in POINTS:
            cv2.circle(img, p, 6, (0, 0, 255), -1)
        if len(POINTS) > 1:
            cv2.polylines(img, [__import__("numpy").array(POINTS, dtype="int32")],
                          False, (0, 0, 255), 2)
        cv2.imshow(param["win"], img)


def main():
    if len(sys.argv) < 2:
        print("用法: python pick_roi.py 图片路径")
        return

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"读不到图片: {path}")
        return

    H, W = img.shape[:2]
    scale = 1280 / W if W > 1280 else 1.0
    disp = cv2.resize(img, None, fx=scale, fy=scale)

    win = "pick roi - left click points, SPACE to finish, ESC to cancel"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click, {"img": disp, "win": win})
    cv2.imshow(win, disp)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:                      # ESC
            print("已取消")
            cv2.destroyAllWindows()
            return
        if k in (13, 32):                # Enter / Space
            break

    cv2.destroyAllWindows()

    if len(POINTS) < 3:
        print(f"只点了 {len(POINTS)} 个点，至少要 3 个")
        return

    # 显示坐标 → 原图归一化坐标（保留 3 位小数）
    norm = [(x / disp.shape[1], y / disp.shape[0]) for x, y in POINTS]
    roi = " ".join(f"{x:.3f},{y:.3f}" for x, y in norm)

    print("\n===== 复制下面这一行，粘到命令里 =====")
    print(f'--roi "{roi}"')
    print("\n完整命令示例：")
    print(f'python demo_v4.py --source "{path}" --model D:\\models\\ebike26.pt '
          f'--roi "{roi}" --out D:\\datasets2\\demo_v4')
    print(f"\n（共 {len(POINTS)} 个点）")


if __name__ == "__main__":
    main()
