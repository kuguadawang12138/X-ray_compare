# -*- coding: utf-8 -*-
"""
方案三：感知哈希（pHash / aHash / dHash，无需对齐）

思路：把图缩成很小并提取“指纹(哈希)”，用汉明距离比较两张图的指纹相似度。
     速度极快、对亮度/轻微缩放鲁棒。
重要局限：标准感知哈希【不抗旋转】——图歪了哈希会大幅变化。
         所以对“每次拍摄角度不一样”的场景，pHash 单独用并不合适，
         这里跑出来正好让你直观看到它的局限。
输出：results_phash —— 两图缩略 + 各哈希的相似度
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, "photos")
OUT_DIR = os.path.join(BASE_DIR, "results_phash")
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


def phash(gray):
    """DCT 感知哈希，返回 64 位布尔指纹。"""
    img = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img))
    low = dct[:8, :8]
    med = np.median(low[1:, 1:])   # 排除直流分量后取中位数
    return (low > med).flatten()


def ahash(gray):
    """均值哈希，返回 64 位布尔指纹。"""
    img = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return (img > img.mean()).flatten()


def dhash(gray):
    """差值哈希，返回 64 位布尔指纹。"""
    img = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (img[:, 1:] > img[:, :-1]).flatten()


def sim_of(h1, h2):
    """汉明距离 -> 相似度(0~1)。"""
    hamming = int(np.count_nonzero(h1 != h2))
    return 1.0 - hamming / len(h1), hamming


def compare(name):
    base_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}_ori.jpg"))
    test_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}.jpg"))
    if base_bgr is None or test_bgr is None:
        print(f"[跳过] 第{name}组：读取失败")
        return
    base_gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    sp, hp = sim_of(phash(base_gray), phash(test_gray))
    sa, ha = sim_of(ahash(base_gray), ahash(test_gray))
    sd, hd = sim_of(dhash(base_gray), dhash(test_gray))

    # 可视化：两图等高缩略并排
    H = 300
    def fit(img):
        scale = H / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), H))
    bcol, tcol = fit(base_bgr), fit(test_bgr)
    gap = np.full((H, 10, 3), 255, np.uint8)
    combo = np.hstack([bcol, gap, tcol])
    bar = np.full((78, combo.shape[1], 3), 255, np.uint8)
    bar = put_text_cn(bar, f"[感知哈希] 第{name}组", (10, 6), 22, (0, 0, 0))
    bar = put_text_cn(bar, f"pHash相似度={sp:.2%}(汉明{hp})  aHash={sa:.2%}(汉明{ha})  dHash={sd:.2%}(汉明{hd})",
                      (10, 40), 18, (0, 0, 0))
    combo = put_text_cn(combo, "基准图", (10, H - 30), 22, (0, 0, 255))
    combo = put_text_cn(combo, "待测图", (bcol.shape[1] + 20, H - 30), 22, (0, 0, 255))
    out = np.vstack([bar, combo])

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"phash_{name}.png"), out)
    print(f"[感知哈希] 第{name}组: pHash={sp:.2%}(汉明{hp})  aHash={sa:.2%}(汉明{ha})  dHash={sd:.2%}(汉明{hd})")


def main():
    print("=" * 60)
    print("方案三：感知哈希（无需对齐，但不抗旋转）")
    for name in PAIRS:
        compare(name)
    print("=" * 60)
    print("结果保存在 results_phash 目录")


if __name__ == "__main__":
    main()
