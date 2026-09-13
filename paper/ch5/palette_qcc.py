# -*- coding: utf-8 -*-
"""第五章配图统一调色板（2026-09-13 定稿）。

口径：**单色相 + 高级感参数**——明度为主轴（OKLCH，感知均匀）、饱和度走钟形
（浅端 10%、中段 100%、深端 82%）、色相在 ±3° 内微移（浅端偏暖、深端偏冷）。
只用一种色相变明暗，因此矩阵里"值大的格子"不会与其余格子跳色；明度承载全部
信号，故过灰度打印与色盲检验都成立。

已定用法：
  - 图 2a  ETTh1-96   → 赭褐 TAUPE  (H=45)
  - 图 2b  ETTm2-96   → 绛红 OXBLOOD(H=18)
  - 图 2d  ETTh1-720  → 墨紫 PLUM   (H=315)
  - 图 1   winheat    → 深青 TEAL   (H=195)   （唯一冷色，与三张暖色 K 图分开）
"""
import math

HUES = {"墨蓝": 252, "深青": 195, "松绿": 152, "赭褐": 45, "绛红": 18, "墨紫": 315}

# 浅端 10% → 中段 100% → 深端 82%（钟形，见设计规范）
_CHROMA_PROFILE = [0.10, 0.32, 0.52, 0.72, 0.90, 1.00, 0.99, 0.93, 0.82]


def oklch_to_srgb(L, C, H):
    """OKLCH → sRGB（0–1）。"""
    a = C * math.cos(math.radians(H))
    b = C * math.sin(math.radians(H))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    f = lambda u: 12.92 * u if u <= 0.0031308 else 1.055 * (u ** (1 / 2.4)) - 0.055
    return tuple(min(1.0, max(0.0, f(u))) for u in (r, g, bb))


def _hex(c):
    return "#%02X%02X%02X" % tuple(int(round(u * 255)) for u in c)


def ramp(hue, n=9, c_peak=0.075, L_hi=0.965, L_lo=0.270, shift=3.0):
    """生成 n 级单色相色带（列表，浅→深）。hue 可传名字（"赭褐"）或角度。"""
    if isinstance(hue, str):
        hue = HUES[hue]
    out = []
    for i in range(n):
        t = i / (n - 1)
        ts = t * t * (3 - 2 * t)                      # smoothstep：中间步距更密
        L = L_hi + (L_lo - L_hi) * ts
        # 把钟形曲线重采样到 n 级
        x = t * (len(_CHROMA_PROFILE) - 1)
        i0 = min(int(x), len(_CHROMA_PROFILE) - 2)
        w = x - i0
        prof = _CHROMA_PROFILE[i0] * (1 - w) + _CHROMA_PROFILE[i0 + 1] * w
        out.append(_hex(oklch_to_srgb(L, c_peak * prof, hue - shift * (t - 0.5) * 2)))
    return out


def cmap(hue, n=256):
    """给 matplotlib 用的连续色图（浅→深）。"""
    from matplotlib.colors import LinearSegmentedColormap
    name = hue if isinstance(hue, str) else f"H{hue}"
    return LinearSegmentedColormap.from_list(f"qcc_{name}", ramp(hue, n=n))


def text_on(bg):
    """按 WCAG 对比度选字色（谁对比高用谁），比固定亮度阈值敏感。"""
    def f(u):
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    L = lambda c: 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])
    def ratio(a, b):
        x, y = L(a), L(b)
        hi, lo = max(x, y), min(x, y)
        return (hi + 0.05) / (lo + 0.05)
    return "white" if ratio((1, 1, 1), bg) >= ratio((0.11, 0.11, 0.11), bg) else "#1C1C1C"


# 统一规格（C3 风格）
GRID_LW   = 1.6          # 格线粗细
NUM_FS    = 9            # 格内数字字号
DIAG_BG   = "#F0F0F0"    # 对角（隐藏）纯浅灰，无纹理
