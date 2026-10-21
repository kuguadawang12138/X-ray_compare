# -*- coding: utf-8 -*-
"""
方案二：Hu 不变矩（无需对齐）

思路：Hu 矩是 7 个对【旋转/缩放/平移/镜像】都不变的形状描述子。
     把零件外形提取出来后，用 cv2.matchShapes(基于Hu矩) 比较两个形状的差异，
     得到一个整体相似度。天然抗旋转，不需要把图摆正。
注意：Hu 矩只描述“整体形状轮廓”，非常粗，无法定位局部差异，
     且依赖前景分割质量；适合做整体形状是否一致的快速判断。
输出：results_hu —— 两图二值剪影 + 相似度
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, "photos")
OUT_DIR = os.path.join(BASE_DIR, "results_hu")
PAIRS = ["1", "2", "3"]
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def put_text_cn(img_bgr, text, org, font_size=24, color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_shape(gray):
    """提取零件二值剪影并返回(二值图, 最大轮廓)。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None
    return th, cnt


def log_hu(cnt):
    """计算并对数缩放 7 个 Hu 矩，便于阅读。"""
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    return [-1 * np.copysign(1.0, h) * np.log10(abs(h) + 1e-30) for h in hu]


def compare(name):
    base_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}_ori.jpg"))
    test_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}.jpg"))
    if base_bgr is None or test_bgr is None:
        print(f"[跳过] 第{name}组：读取失败")
        return
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    base_th, base_cnt = get_shape(base_gray)
    test_th, test_cnt = get_shape(test_gray)
    if base_cnt is None or test_cnt is None:
        print(f"[跳过] 第{name}组：形状提取失败")
        return

    # matchShapes 基于 Hu 矩，返回值越小越相似（0=完全一致）
    dist = cv2.matchShapes(base_cnt, test_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    sim = 1.0 / (1.0 + dist)        # 转成 0~1 相似度，越大越像

    # 可视化：左基准剪影 | 右待测剪影
    h = max(base_th.shape[0], test_th.shape[0])
    def fit(img):
        c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pad = np.full((h - c.shape[0], c.shape[1], 3), 0, np.uint8)
        return np.vstack([c, pad]) if pad.shape[0] > 0 else c
    bcol, tcol = fit(base_th), fit(test_th)
    gap = np.full((h, 10, 3), 255, np.uint8)
    combo = np.hstack([bcol, gap, tcol])
    bar = np.full((46, combo.shape[1], 3), 255, np.uint8)
    bar = put_text_cn(bar, f"[Hu不变矩] 第{name}组  matchShapes距离={dist:.4f}  相似度={sim:.4f}",
                      (10, 9), 22, (0, 0, 0))
    combo = put_text_cn(combo, "基准剪影", (10, h - 30), 22, (0, 255, 0))
    combo = put_text_cn(combo, "待测剪影", (bcol.shape[1] + 20, h - 30), 22, (0, 255, 0))
    out = np.vstack([bar, combo])

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"hu_{name}.png"), out)
    bh = log_hu(base_cnt); th_ = log_hu(test_cnt)
    print(f"[Hu] 第{name}组: 距离={dist:.4f}  相似度={sim:.4f}")
    print(f"     基准logHu={[round(v,2) for v in bh]}")
    print(f"     待测logHu={[round(v,2) for v in th_]}")


def main():
    print("=" * 60)
    print("方案二：Hu 不变矩（无需对齐，抗旋转）")
    for name in PAIRS:
        compare(name)
    print("=" * 60)
    print("结果保存在 results_hu 目录")


if __name__ == "__main__":
    main()
