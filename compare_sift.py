# -*- coding: utf-8 -*-
"""
方案一：SIFT 特征匹配（无需数据预处理/对齐）

思路：直接在原图灰度上提取 SIFT 特征点并匹配，用“匹配点数量/几何一致内点数”
     来衡量两图相似度。SIFT 对旋转、缩放天然不变，因此不需要把图摆正，
     适合“每次拍摄角度不一样”的场景。
输出：results_sift —— 匹配连线可视化 + 相似度指标
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, "photos")
OUT_DIR = os.path.join(BASE_DIR, "results_sift")
PAIRS = ["1", "2", "3"]
RATIO = 0.75                      # Lowe 比率检验阈值
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def put_text_cn(img_bgr, text, org, font_size=24, color=(0, 0, 0)):
    """用 PIL 绘制中文，避免 OpenCV putText 乱码。"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def compare(name):
    base_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}_ori.jpg"))
    test_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}.jpg"))
    if base_bgr is None or test_bgr is None:
        print(f"[跳过] 第{name}组：读取失败")
        return
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(test_gray, None)
    kp2, d2 = sift.detectAndCompute(base_gray, None)

    good = []
    if d1 is not None and d2 is not None and len(kp1) > 2 and len(kp2) > 2:
        bf = cv2.BFMatcher()
        raw = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in raw if m.distance < RATIO * n.distance]

    inliers = 0
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

    n_min = max(min(len(kp1), len(kp2)), 1)
    sim_match = len(good) / n_min          # 匹配覆盖率：好匹配/较少一方特征点数
    sim_inlier = inliers / max(len(good), 1)  # 几何一致率：内点/好匹配

    # 可视化：画前 40 个好匹配的连线
    vis = cv2.drawMatches(test_bgr, kp1, base_bgr, kp2, good[:40], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    bar = np.full((46, vis.shape[1], 3), 255, np.uint8)
    txt = (f"[SIFT] 第{name}组  特征点(待测{len(kp1)}/基准{len(kp2)})  "
           f"好匹配={len(good)}  内点={inliers}  匹配覆盖率={sim_match:.2%}  几何一致率={sim_inlier:.2%}")
    bar = put_text_cn(bar, txt, (10, 9), 20, (0, 0, 0))
    out = np.vstack([bar, vis])

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"sift_{name}.png"), out)
    print(f"[SIFT] 第{name}组: 特征点 待测{len(kp1)}/基准{len(kp2)}  好匹配={len(good)}  "
          f"内点={inliers}  匹配覆盖率={sim_match:.2%}  几何一致率={sim_inlier:.2%}")


def main():
    print("=" * 60)
    print("方案一：SIFT 特征匹配（无预处理）")
    for name in PAIRS:
        compare(name)
    print("=" * 60)
    print("结果保存在 results_sift 目录")


if __name__ == "__main__":
    main()
