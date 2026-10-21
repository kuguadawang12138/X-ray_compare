# -*- coding: utf-8 -*-
"""
X光图像对比脚本（方向A：同种零件·找局部缺陷）

与方向B(compare_images.py)的区别只在“差异判定的假设”：
  - A 假设待测图与基准图本该基本一致，因此使用【高容差 + 高阈值】，
    忽略大部分可被微小对齐误差解释的差异，只揪出少数几个明显的局部缺陷。
  - 预处理/对齐/相似度的逻辑与 B 完全相同（共用同一套对齐，保证可比性）。
输出目录：results_A
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

# ---------------- 配置 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, "photos")
OUT_DIR = os.path.join(BASE_DIR, "results_A")
# 方向A(局部缺陷检测)：高容差、高阈值，只标少数明显缺陷
PAIRS = ["1", "2", "3"]          # 三组图片
DIFF_THRESH = 78                 # 差异灵敏度阈值(0~255)，A取高值，只抓强差异
TOLERANCE_BLUR = 4.0             # 容差对比的高斯模糊sigma，A取大值，尽量忽略对齐误差
# —— 网格切块定位：格子尺寸按零件(前景)短边比例自适应，大零件框大、小零件框小 ——
CELL_SIZE_RATIO = 0.18           # 框尺寸占零件短边的比例（调小→框更小更具体，调大→框更大）
CELL_DIFF_RATIO = 0.15           # 一个格子内差异像素占比超过此值才框出（调小→更灵敏、框更多）
TOP_N_BOXES = 12                 # 最多框出的差异格子数（按差异强度从大到小）
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
# --------------------------------------


def put_text_cn(img_bgr, text, org, font_size=24, color=(0, 0, 0)):
    """用 PIL 在图像上绘制中文，避免 OpenCV putText 中文乱码。"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def remove_red_watermark(img_bgr):
    """去除红色水印：检测红色像素并用周边内容修复(inpaint)。
    返回：(修复后图像, 水印掩膜)。水印掩膜用于后续差异检测时排除该区域。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 80]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80]); upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    if cv2.countNonZero(mask) == 0:
        return img_bgr, mask
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    fixed = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
    return fixed, mask


def foreground_bbox(gray):
    """分割元件前景(比浅色背景暗)，返回最大连通区域的外接框 (x,y,w,h)。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = gray.shape
        return (0, 0, w, h)
    return cv2.boundingRect(max(cnts, key=cv2.contourArea))


def align_image(test_gray, base_gray):
    """
    将待测图对齐到基准图坐标系。
    步骤：① 用前景外接框做等比缩放+平移(对齐中心与尺寸)
         ② 用 ECC(欧氏变换:旋转+平移)微调小角度旋转
    返回：对齐后灰度图(尺寸同基准)、有效区域掩膜、对齐方式说明。
    """
    H, W = base_gray.shape

    bx, by, bw, bh = foreground_bbox(base_gray)
    tx, ty, tw, th = foreground_bbox(test_gray)

    # ① 等比缩放(取宽高比例均值，保证不拉伸) + 中心平移
    s = ((bw / max(tw, 1)) + (bh / max(th, 1))) / 2.0
    s = float(np.clip(s, 0.2, 5.0))
    tcx, tcy = tx + tw / 2.0, ty + th / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    M = np.array([[s, 0, bcx - s * tcx],
                  [0, s, bcy - s * tcy]], dtype=np.float32)
    pre = cv2.warpAffine(test_gray, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpAffine(np.full(test_gray.shape, 255, np.uint8), M, (W, H),
                           flags=cv2.INTER_NEAREST, borderValue=0)
    how = f"轮廓对齐(缩放×{s:.3f})"

    # ② ECC 微调旋转+平移（用有效区掩膜，并对图像做高斯平滑帮助收敛）
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        ecc_mask = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        cc, warp = cv2.findTransformECC(base_gray, pre, warp,
                                        cv2.MOTION_EUCLIDEAN, criteria, ecc_mask, 5)
        rot = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
        if abs(rot) < 25:  # 仅接受小角度修正，避免异常旋转
            pre = cv2.warpAffine(pre, warp, (W, H),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderValue=0)
            valid = cv2.warpAffine(valid, warp, (W, H),
                                   flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP, borderValue=0)
            how += f"+ECC旋转{rot:.1f}°"
    except Exception:
        how += "+ECC未收敛(略)"

    return pre, valid, how


def normalize_brightness(t, base, valid_bool):
    """线性亮度归一化：把 t 在有效区的均值/方差对齐到 base，避免色阶断层。"""
    tm, ts = t[valid_bool].mean(), t[valid_bool].std() + 1e-6
    bm, bs = base[valid_bool].mean(), base[valid_bool].std() + 1e-6
    out = (t.astype(np.float32) - tm) / ts * bs + bm
    return np.clip(out, 0, 255).astype(np.uint8)


def find_diff_boxes(thd, cell, min_ratio, top_n):
    """网格切块定位差异：把二值差异图按 cell 大小切成小格子，
    格子内差异像素占比超过 min_ratio 的就框出来（框收紧到实际差异像素）。
    返回按差异强度从大到小排序、最多 top_n 个的小框列表 [(x,y,w,h),...]。
    这样每个框最大不超过一个格子，避免一大片连续差异只得到一个大框。"""
    H, W = thd.shape
    cell = max(int(cell), 8)
    boxes = []
    for y in range(0, H, cell):
        for x in range(0, W, cell):
            sub = thd[y:y + cell, x:x + cell]
            nz = int(cv2.countNonZero(sub))
            if nz >= min_ratio * sub.size and nz > 0:
                ys, xs = np.where(sub > 0)
                bx, by = x + int(xs.min()), y + int(ys.min())
                bw = int(xs.max() - xs.min()) + 1
                bh = int(ys.max() - ys.min()) + 1
                boxes.append((bx, by, bw, bh, nz))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return [b[:4] for b in boxes[:top_n]]


def compare_pair(name):
    base_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}_ori.jpg"))
    test_bgr = cv2.imread(os.path.join(PHOTO_DIR, f"{name}.jpg"))
    if base_bgr is None or test_bgr is None:
        print(f"[跳过] 第{name}组：图片读取失败")
        return

    base_clean, base_wm = remove_red_watermark(base_bgr)
    test_clean, _ = remove_red_watermark(test_bgr)
    base_gray = cv2.cvtColor(base_clean, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_clean, cv2.COLOR_BGR2GRAY)
    # 水印区域较大膨胀，后续差异检测时排除，避免水印残留被误判为差异
    if cv2.countNonZero(base_wm) > 0:
        base_wm = cv2.dilate(base_wm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                             iterations=2)

    aligned_gray, valid_mask, how = align_image(test_gray, base_gray)
    valid_bool = valid_mask > 0
    if valid_bool.sum() < 100:
        print(f"[跳过] 第{name}组：有效重叠区过小")
        return

    aligned_norm = normalize_brightness(aligned_gray, base_gray, valid_bool)
    H, W = base_gray.shape

    # 头条相似度：在有效区用原始(未模糊)灰度算 SSIM，反映真实结构相似度
    score_full, diff_struct = ssim(base_gray, aligned_norm, full=True)
    score = float(diff_struct[valid_bool].mean())

    # 容差对比：A方向用较大模糊，尽量忽略对齐误差，只留明显缺陷
    b_blur = cv2.GaussianBlur(base_gray, (0, 0), TOLERANCE_BLUR)
    t_blur = cv2.GaussianBlur(aligned_norm, (0, 0), TOLERANCE_BLUR)
    _, diff_map = ssim(b_blur, t_blur, full=True)

    diff_u8 = ((1.0 - diff_map) * 255).astype(np.uint8)
    diff_u8[~valid_bool] = 0
    # 适当腐蚀有效区，去掉因对齐误差产生的边缘细线假差异
    er = cv2.erode(valid_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    diff_u8[er == 0] = 0
    # 排除水印区域，避免水印残留被误判为差异
    if cv2.countNonZero(base_wm) > 0:
        diff_u8[base_wm > 0] = 0

    # 高阈值(只抓强差异) + 形态学清噪
    _, thd = cv2.threshold(diff_u8, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thd = cv2.morphologyEx(thd, cv2.MORPH_OPEN, k)

    # 网格切块定位：格子尺寸按零件(前景)短边自适应，实现“按零件大小控制框大小”
    fx, fy, fw, fh = foreground_bbox(base_gray)
    cell = CELL_SIZE_RATIO * min(fw, fh)
    boxes = find_diff_boxes(thd, cell, CELL_DIFF_RATIO, TOP_N_BOXES)

    base_annot = base_clean.copy()
    test_annot = cv2.cvtColor(aligned_norm, cv2.COLOR_GRAY2BGR)
    for i, (x, y, bw, bh) in enumerate(boxes, 1):
        cv2.rectangle(base_annot, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.rectangle(test_annot, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(base_annot, str(i), (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(test_annot, str(i), (x, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    box_count = len(boxes)

    heat = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
    overlay = cv2.cvtColor(aligned_norm, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.6, heat, 0.4, 0)
    overlay[~valid_bool] = (255, 255, 255)

    base_annot = put_text_cn(base_annot, "基准图", (10, H - 34), 22, (0, 0, 255))
    test_annot = put_text_cn(test_annot, "待测图(已对齐)", (10, H - 34), 22, (0, 0, 255))
    overlay = put_text_cn(overlay, "差异热力图", (10, H - 34), 22, (0, 0, 255))

    gap = np.full((H, 10, 3), 255, np.uint8)
    combo = np.hstack([base_annot, gap, test_annot, gap, overlay])
    bar = np.full((44, combo.shape[1], 3), 255, np.uint8)
    bar = put_text_cn(bar, f"[方向A] 第{name}组   相似度(SSIM)={score:.4f}   明显缺陷={box_count}处   {how}",
                      (10, 8), font_size=24, color=(0, 0, 0))
    out = np.vstack([bar, combo])

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, f"compare_{name}.png"), out)
    cv2.imwrite(os.path.join(OUT_DIR, f"diff_{name}.png"), heat)
    print(f"[方向A] 第{name}组: 相似度(SSIM)={score:.4f}  明显缺陷={box_count}处  对齐={how}")


def main():
    print("=" * 60)
    print("方向A：同种零件·只标少数明显缺陷")
    for name in PAIRS:
        compare_pair(name)
    print("=" * 60)
    print("全部完成，结果保存在 results_A 目录")


if __name__ == "__main__":
    main()
