# -*- coding: utf-8 -*-
"""Generates the hand-drawn-style diagrams (SVG -> PNG) for the reading notes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgrender import render

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)

INK = "#1f2937"; MUTED = "#6b7280"
BLUE = "#2563eb"; BLUE_BG = "#eef4ff"
AMBER = "#b45309"; AMBER_BG = "#fff7e6"
GREEN = "#047857"; GREEN_BG = "#ecfdf5"
PURPLE = "#6d28d9"; PURPLE_BG = "#f5f0ff"
RED = "#b91c1c"; RED_BG = "#fef2f2"
TEAL = "#0f766e"; TEAL_BG = "#ecfeff"
GRAY_BG = "#f8fafc"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h, bg="#ffffff"):
        self.w, self.h, self.bg = w, h, bg
        self.parts = []

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.5, rx=8, opacity=1, fo=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" fill-opacity="{fo}"{d}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1.8, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def path(self, d, stroke="none", fill="none", sw=1.8, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" stroke="{stroke}" fill="{fill}" stroke-width="{sw}"{da}/>')

    def poly(self, pts, fill="none", stroke="none", sw=1.5):
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, cx, cy, r, fill="none", stroke=INK, sw=1.6):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, fs=16, fill=INK, anchor="start", bold=False, rot=0, opacity=1):
        b = ' font-weight="bold"' if bold else ""
        r = f' rot="{rot}"' if rot else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{fs}" fill="{fill}" '
                          f'text-anchor="{anchor}" data-text="{esc(s)}"{b}{r} opacity="{opacity}"/>')

    def arrow_v(self, x, y1, y2, stroke=INK, sw=1.8, head=9, dash=None):
        if y2 < y1:  # upward
            self.line(x, y1, x, y2 + head * 0.9, stroke, sw, dash)
            self.poly([(x, y2), (x - head * 0.55, y2 + head), (x + head * 0.55, y2 + head)], fill=stroke)
        else:
            self.line(x, y1, x, y2 - head * 0.9, stroke, sw, dash)
            self.poly([(x, y2), (x - head * 0.55, y2 - head), (x + head * 0.55, y2 - head)], fill=stroke)

    def arrow_h(self, x1, x2, y, stroke=INK, sw=1.8, head=9, dash=None):
        sgn = 1 if x2 > x1 else -1
        self.line(x1, y, x2 - sgn * head * 0.9, y, stroke, sw, dash)
        self.poly([(x2, y), (x2 - sgn * head, y - head * 0.55),
                   (x2 - sgn * head, y + head * 0.55)], fill=stroke)

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">' + "".join(self.parts) + "</svg>")

    def save(self, name, width=None):
        png = os.path.join(OUT, name)
        render(self.svg(), png, width or self.w, self.h, self.bg)
        print("saved", png)


# =====================================================================
# 图 1：整体结构（标注版）
# =====================================================================
def fig_overview():
    W, H = 1900, 1600
    s = SVG(W, H, "#ffffff")
    s.rect(0, 0, W, H, fill="#ffffff", rx=0)

    s.text(48, 52, "Transformer 结构图（读完《Attention Is All You Need》后自绘 · 标注版）", fs=30, bold=True)
    s.text(48, 88, "左：我从 Figure 1 读出的“信息流骨架”，补了残差 / LayerNorm 的走线；右：Figure 1 没画出来的矩阵形状、多头、mask 和 Q/K/V 来源",
           fs=15, fill=MUTED)
    s.line(48, 104, W - 48, 104, stroke="#e5e7eb", sw=1.5)

    s.text(330, 142, "Encoder 编码器", fs=20, bold=True, fill=BLUE, anchor="middle")
    s.text(330, 166, "N = 6 层，每层 2 个子层", fs=13.5, fill=MUTED, anchor="middle")
    s.text(740, 142, "Decoder 解码器", fs=20, bold=True, fill=AMBER, anchor="middle")
    s.text(740, 166, "N = 6 层，每层 3 个子层", fs=13.5, fill=MUTED, anchor="middle")

    # ================= ENCODER =================
    ex, ew = 130, 400
    cx = ex + ew / 2
    s.rect(ex, 186, ew, 910, fill=GRAY_BG, stroke="#cbd5e1", sw=1.6, rx=16, dash="7 6")
    s.text(ex + ew - 14, 212, "× 6 层（层与层之间形状不变）", fs=12.5, fill=MUTED, anchor="end")

    # --- 子层一：MHSA ---
    y1, h1 = 222, 118
    s.rect(ex + 24, y1, ew - 48, h1, fill=BLUE_BG, stroke=BLUE, sw=2, rx=10)
    s.text(ex + 42, y1 + 27, "Multi-Head Self-Attention", fs=16.5, bold=True, fill=BLUE)
    s.text(ex + 42, y1 + 52, "8 个头，每个头 d_k = d_v = 64", fs=13, fill=INK)
    s.text(ex + 42, y1 + 73, "Q、K、V 全部来自本层输入", fs=12, fill=MUTED)
    s.text(ex + 42, y1 + 94, "→ 用 8 个视角看同一句话", fs=11.5, fill=MUTED)
    for k in range(8):
        s.rect(432 + k * 11, y1 + 16, 9, 14, fill=BLUE, opacity=0.25 + 0.09 * k, rx=2, stroke="none")
    s.text(432, y1 + 44, "head 1 … 8", fs=10, fill=BLUE)

    # --- 子层二：FFN ---
    y2, h2 = 398, 104
    s.rect(ex + 24, y2, ew - 48, h2, fill=AMBER_BG, stroke=AMBER, sw=2, rx=10)
    s.text(ex + 42, y2 + 27, "Position-wise Feed-Forward", fs=16.5, bold=True, fill=AMBER)
    s.text(ex + 42, y2 + 52, "Linear(512→2048) → ReLU → Linear(2048→512)", fs=12, fill=INK)
    s.text(ex + 42, y2 + 74, "对每个位置独立作用、层内共享参数", fs=12, fill=MUTED)

    s.arrow_v(cx, y1 + h1, y2, stroke=INK, sw=2)

    # 残差 + LayerNorm（画在子层出口）
    for top, bot in ((y1 + 4, y1 + h1 + 6), (y2 + 4, y2 + h2 + 6)):
        s.path(f"M {ex + 50} {bot} C {ex + 2} {bot} {ex + 2} {top} {ex + 50} {top}",
               stroke=PURPLE, sw=2, dash="6 5")
        s.circle(ex + 50, bot, 9.5, fill="#ffffff", stroke=PURPLE, sw=2)
        s.text(ex + 50, bot + 5.5, "+", fs=15, fill=PURPLE, anchor="middle", bold=True)
        s.rect(ex + 68, bot - 13, 92, 26, fill=GREEN_BG, stroke=GREEN, sw=1.8, rx=7)
        s.text(ex + 114, bot + 4.5, "LayerNorm", fs=12, fill=GREEN, anchor="middle", bold=True)

    # 层堆叠示意
    s.arrow_v(cx, y2 + h2 + 10, 590, stroke=INK, sw=2)
    s.text(cx + 150, 534, "第 2 层", fs=13, fill=MUTED)
    s.text(cx + 150, 560, "⋮", fs=15, fill=MUTED)
    s.arrow_v(cx, 600, 672, stroke=INK, sw=2)
    s.text(cx + 150, 640, "第 6 层", fs=13, fill=MUTED)

    # --- 底部：token + 嵌入 + 位置编码 ---
    s.rect(ex + 60, 1036, ew - 120, 44, fill="#ffffff", stroke=INK, sw=1.6, rx=8)
    s.text(cx, 1064, "输入 token 序列  x1  x2  …  xn", fs=14, anchor="middle")
    s.arrow_v(cx, 1036, 1006, stroke=INK, sw=2)

    s.rect(ex, 792, ew, 214, fill="#ffffff", stroke=PURPLE, sw=1.8, rx=12, dash="6 5")
    s.rect(ex + 24, 806, ew - 48, 48, fill=BLUE_BG, stroke=BLUE, sw=1.6, rx=8)
    s.text(cx, 836, "Input Embedding（每个词 → 512 维向量）", fs=13, anchor="middle", fill=BLUE)
    s.arrow_v(cx, 806, 786, stroke=INK, sw=2)
    s.circle(cx, 884, 11, fill="#ffffff", stroke=PURPLE, sw=2)
    s.text(cx, 889.5, "+", fs=17, fill=PURPLE, anchor="middle", bold=True)
    s.rect(ex + 24, 918, ew - 48, 44, fill=PURPLE_BG, stroke=PURPLE, sw=1.6, rx=8)
    s.text(cx, 946, "Positional Encoding（正弦/余弦，与嵌入同维）", fs=12, anchor="middle", fill=PURPLE)
    s.arrow_v(cx, 918, 896, stroke=PURPLE, sw=1.6)
    s.arrow_v(cx, 872, 858, stroke=INK, sw=1.8)
    s.text(cx + 16, 868, "逐元素相加", fs=11, fill=MUTED)
    s.arrow_v(cx, 792, 676, stroke=INK, sw=2)

    s.text(cx, 1146, "所有子层（含嵌入）输出维度都是 d_model = 512", fs=13, fill=MUTED, anchor="middle")
    s.text(cx, 1170, "⇒ 残差连接可以直接逐元素相加（图上的 + 号）", fs=13, fill=MUTED, anchor="middle")

    # ================= DECODER =================
    dx, dw = 540, 400
    dcx = dx + dw / 2
    s.rect(dx, 186, dw, 910, fill="#fffdf7", stroke="#d6c7a1", sw=1.6, rx=16, dash="7 6")
    s.text(dx + 14, 212, "× 6 层（自回归：一次只喂已生成的前缀）", fs=12.5, fill=MUTED)

    # --- 子层一：masked self-attn ---
    d1y, d1h = 220, 112
    s.rect(dx + 24, d1y, dw - 48, d1h, fill=RED_BG, stroke=RED, sw=2, rx=10)
    s.text(dx + 42, d1y + 27, "Masked Multi-Head Self-Attention", fs=15.5, bold=True, fill=RED)
    s.text(dx + 42, d1y + 52, "结构同左侧，但每个 query 被限制：", fs=12, fill=INK)
    s.text(dx + 42, d1y + 72, "位置 i 只能看见位置 ≤ i（不能看未来）", fs=12, fill=MUTED)
    s.text(dx + 42, d1y + 92, "→ 因果 mask，见右侧 ③", fs=11, fill=MUTED)

    # --- 子层二：cross-attn ---
    d2y, d2h = 350, 108
    s.rect(dx + 24, d2y, dw - 48, d2h, fill=TEAL_BG, stroke=TEAL, sw=2, rx=10)
    s.text(dx + 42, d2y + 27, "Cross-Attention（Encoder-Decoder Attn）", fs=15, bold=True, fill=TEAL)
    s.text(dx + 42, d2y + 52, "Q ← 解码器当前层（我在生成哪个词）", fs=12, fill=INK)
    s.text(dx + 42, d2y + 72, "K, V ← 编码器最终输出（整句原文）", fs=12, fill=TEAL)
    s.text(dx + 42, d2y + 92, "这里不设 mask：源句每个词都可看", fs=11, fill=MUTED)

    # --- 子层三：FFN ---
    d3y, d3h = 476, 104
    s.rect(dx + 24, d3y, dw - 48, d3h, fill=AMBER_BG, stroke=AMBER, sw=2, rx=10)
    s.text(dx + 42, d3y + 27, "Position-wise Feed-Forward", fs=16.5, bold=True, fill=AMBER)
    s.text(dx + 42, d3y + 52, "Linear(512→2048) → ReLU → Linear(2048→512)", fs=12, fill=INK)
    s.text(dx + 42, d3y + 74, "位置之间仍然互不交流", fs=12, fill=MUTED)

    s.arrow_v(dcx, d1y + d1h, d2y, stroke=INK, sw=2)
    s.arrow_v(dcx, d2y + d2h, d3y, stroke=INK, sw=2)

    for top, bot in ((d1y + 4, d1y + d1h + 6), (d2y + 4, d2y + d2h + 6), (d3y + 4, d3y + d3h + 6)):
        s.path(f"M {dx + 48} {bot} C {dx + 2} {bot} {dx + 2} {top} {dx + 48} {top}",
               stroke=PURPLE, sw=2, dash="6 5")
        s.circle(dx + 48, bot, 9.5, fill="#ffffff", stroke=PURPLE, sw=2)
        s.text(dx + 48, bot + 5.5, "+", fs=15, fill=PURPLE, anchor="middle", bold=True)
        s.rect(dx + 66, bot - 13, 92, 26, fill=GREEN_BG, stroke=GREEN, sw=1.8, rx=7)
        s.text(dx + 112, bot + 4.5, "LayerNorm", fs=12, fill=GREEN, anchor="middle", bold=True)

    s.arrow_v(dcx, d3y + d3h + 10, 668, stroke=INK, sw=2)
    s.text(dcx + 14, 614, "第 6 层输出", fs=12.5, fill=MUTED)

    # --- 输出嵌入 ---
    s.rect(dx, 806, dw, 200, fill="#ffffff", stroke=PURPLE, sw=1.8, rx=12, dash="6 5")
    s.rect(dx + 24, 820, dw - 48, 50, fill=AMBER_BG, stroke=AMBER, sw=1.6, rx=8)
    s.text(dcx, 843, "Output Embedding", fs=13.5, anchor="middle", fill=AMBER)
    s.text(dcx, 862, "（右移一位的已生成序列）", fs=11, anchor="middle", fill=AMBER)
    s.arrow_v(dcx, 820, 796, stroke=INK, sw=2)
    s.circle(dcx, 894, 11, fill="#ffffff", stroke=PURPLE, sw=2)
    s.text(dcx, 899.5, "+", fs=17, fill=PURPLE, anchor="middle", bold=True)
    s.rect(dx + 24, 924, dw - 48, 42, fill=PURPLE_BG, stroke=PURPLE, sw=1.6, rx=8)
    s.text(dcx, 951, "Positional Encoding（同一套公式）", fs=12, anchor="middle", fill=PURPLE)
    s.arrow_v(dcx, 924, 906, stroke=PURPLE, sw=1.6)
    s.arrow_v(dcx, 882, 868, stroke=INK, sw=1.8)
    s.arrow_v(dcx, 806, 674, stroke=INK, sw=2)

    # --- 输出头 ---
    s.rect(dx, 734, dw, 46, fill="#ffffff", stroke=INK, sw=1.6, rx=8)
    s.text(dcx, 764, "Linear（512 → 词表大小）", fs=14, anchor="middle")
    s.arrow_v(dcx, 734, 706, stroke=INK, sw=2)
    s.rect(dx, 652, dw, 46, fill=RED_BG, stroke=RED, sw=1.8, rx=8)
    s.text(dcx, 682, "Softmax → 下一个词的概率分布", fs=14, anchor="middle", fill=RED)
    s.arrow_v(dcx, 652, 628, stroke=INK, sw=2)
    # 输出反馈到输入（自回归）
    s.path(f"M {dx + dw - 4} 675 C {dx + dw + 30} 675 {dx + dw + 30} 742 {dx + dw + 2} 742",
           stroke=RED, sw=1.6, dash="5 4")
    s.text(dx + dw + 34, 700, "输出 token", fs=10.5, fill=RED)
    s.text(dx + dw + 34, 715, "（训练时并行）", fs=10.5, fill=RED)

    # 编码器 → 交叉注意力
    s.path(f"M {ex + ew + 2} 640 C {ex + ew + 40} 640 {dx - 40} 400 {dx + 20} 400",
           stroke=TEAL, sw=2.2, dash="8 5")
    s.arrow_h(dx - 42, dx + 20, 400, stroke=TEAL, sw=2.2)
    s.text(dx - 40, 386, "memory", fs=10, fill=TEAL, anchor="middle")

    # ================= 右侧面板 =================
    px, pw = 1080, 780

    # ① 公式
    s.rect(px, 120, pw, 274, fill="#ffffff", stroke="#d1d5db", sw=1.6, rx=14)
    s.text(px + 22, 152, "① 四条公式与张量形状（论文 3.1–3.3 的一句话版本）", fs=17.5, bold=True, fill=INK)
    rows = [
        ("Scaled Dot-Product", "Attention(Q, K, V) = softmax(QKᵀ/√d_k)·V",
         "Q: n×d_k    K: m×d_k    V: m×d_v  →  输出 n×d_v", BLUE),
        ("Multi-Head", "MultiHead = Concat(head₁ … head₈)·W^O",
         "每个 head 64 维；8 × 64 = 512 = d_model", PURPLE),
        ("Feed-Forward", "FFN(x) = max(0, xW₁+b₁)·W₂+b₂",
         "逐位置作用：512 → 2048 → 512", AMBER),
        ("Positional Encoding", "PE(pos,2i) = sin(pos/10000^{2i/512})",
         "与 cos 交替，直接加在嵌入上（不是拼接）", GREEN),
    ]
    y = 176
    for name, f1, f2, col in rows:
        s.rect(px + 20, y, pw - 40, 50, fill="#fbfdff", stroke="#e5ecf6", sw=1.2, rx=8)
        s.text(px + 34, y + 20, name, fs=12.5, bold=True, fill=col)
        s.text(px + 34, y + 40, f1, fs=13, fill=INK)
        s.text(px + 400, y + 20, f2, fs=11.5, fill=MUTED)
        y += 55

    # ② 多头
    s.rect(px, 414, pw, 214, fill="#ffffff", stroke="#d1d5db", sw=1.6, rx=14)
    s.text(px + 22, 446, "② 8 个头 ≠ 重复 8 次：同一输入 → 8 套投影 → 8 个“观察视角”", fs=17, bold=True, fill=PURPLE)
    s.rect(px + 34, 464, 52, 26, fill="#dbeafe", stroke=MUTED, sw=1.2, rx=5)
    s.text(px + 60, 482, "X", fs=13, anchor="middle", bold=True)
    s.arrow_h(px + 90, px + 196, 477, stroke=MUTED, sw=1.5)
    s.text(px + 206, 481, "8 套互不共享的 W_i^Q / W_i^K / W_i^V（各自 512→64）", fs=12, fill=MUTED)
    hy = 512
    for k in range(8):
        hx = px + 34 + k * 64
        s.rect(hx, hy, 54, 40, fill=PURPLE, opacity=0.13 + 0.05 * k, stroke=PURPLE, sw=1.2, rx=6)
        s.text(hx + 27, hy + 26, f"h{k+1}", fs=12.5, anchor="middle", fill=PURPLE, bold=True)
    s.text(px + 34, hy + 60, "每个头 64 维：h1 可能盯“主谓一致”，h2 盯“指代”，h3 盯“局部搭配”……", fs=11.5, fill=MUTED)
    s.text(px + 34, hy + 82, "Concat → 512 维 → × W^O 融合成一个输出（只有单头就会把这些视角平均掉）", fs=12, fill=INK)

    # ③ mask
    s.rect(px, 644, pw, 300, fill="#ffffff", stroke="#d1d5db", sw=1.6, rx=14)
    s.text(px + 22, 676, "③ Masked Attention：把“未来”的分数设成 −∞，softmax 后权重恰好为 0", fs=16.5, bold=True, fill=RED)
    s.text(px + 22, 700, "左：QKᵀ/√d_k 的分数矩阵（加 mask 前）     右：逐行 softmax 后的注意力权重（每行和为 1）", fs=11.5, fill=MUTED)
    n, cell = 6, 26
    for tag, ox, masked in (("加 mask 前", px + 46, False), ("过 softmax 后", px + 330, True)):
        s.text(ox + 44, 728, tag, fs=12, bold=True, fill=RED)
        for c in range(n):
            s.text(ox + 46 + c * cell + 11, 746, f"k{c+1}", fs=9, fill=MUTED, anchor="middle")
        for r in range(n):
            s.text(ox + 40, 776 + r * cell, f"q{r+1}", fs=9, fill=MUTED, anchor="end")
            for c in range(n):
                x0, y0 = ox + 46 + c * cell, 758 + r * cell
                bad = c > r
                if masked:
                    s.rect(x0, y0, cell - 3, cell - 3, fill="#e5e7eb" if bad else RED,
                           fo=1 if bad else 0.18 + 0.11 * (r - c), stroke="#d1d5db", sw=0.8, rx=3)
                    if bad:
                        s.text(x0 + 11, y0 + 17, "0", fs=9.5, fill=MUTED, anchor="middle")
                else:
                    s.rect(x0, y0, cell - 3, cell - 3, fill=RED_BG if bad else "#fef3c7",
                           stroke="#d1d5db", sw=0.8, rx=3)
                    if bad:
                        s.text(x0 + 11, y0 + 17, "−∞", fs=8, fill=RED, anchor="middle")
        s.text(ox + 44 + n * cell / 2, 758 + n * cell + 18, "行 = query 位置   列 = key 位置", fs=10, fill=MUTED, anchor="middle")
    s.text(px + 520, 786, "0 = 该位置被切断", fs=10.5, fill=MUTED)
    s.text(px + 520, 804, "（exp 和为 0，梯度 0）", fs=10.5, fill=MUTED)
    s.text(px + 520, 838, "对角线及以下：可见", fs=11, fill=GREEN)
    s.text(px + 520, 858, "对角线以上：未来，禁止", fs=11, fill=RED)

    # ④ 三种注意力
    s.rect(px, 960, pw, 448, fill="#ffffff", stroke="#d1d5db", sw=1.6, rx=14)
    s.text(px + 22, 992, "④ 三种 Multi-Head Attention 的唯一区别：Q / K / V 从哪来（论文 3.2.3）", fs=16.5, bold=True, fill=INK)
    cards = [
        ("编码器自注意力", BLUE, "Q ← 上一层编码器输出", "K, V ← 上一层编码器输出",
         "全可见（无 mask）", "每个源词看整句，建立上下文"),
        ("解码器 Masked 自注意力", RED, "Q ← 上一层解码器输出", "K, V ← 上一层解码器输出",
         "只能看 ≤ i（因果 mask）", "维持自回归，不偷看答案"),
        ("编码器-解码器交叉注意力", TEAL, "Q ← 上一层解码器输出", "K, V ← 编码器最终输出",
         "全可见（源句整体）", "对齐原文，像 seq2seq 的 attention"),
    ]
    cy = 1008
    for name, col, l1, l2, l3, l4 in cards:
        s.rect(px + 20, cy, pw - 40, 118, fill="#fbfdff", stroke="#e5e7eb", sw=1.2, rx=9)
        s.rect(px + 20, cy, 5, 118, fill=col, stroke="none", rx=2)
        s.text(px + 38, cy + 26, name, fs=14, bold=True, fill=col)
        s.text(px + 38, cy + 50, l1, fs=12, fill=INK)
        s.text(px + 38, cy + 70, l2, fs=12, fill=INK)
        s.text(px + 400, cy + 50, l3, fs=12, fill=col)
        s.text(px + 400, cy + 70, "作用：" + l4, fs=12, fill=MUTED)
        cy += 126

    # 图例
    s.line(48, 1526, W - 48, 1526, stroke="#e5e7eb", sw=1.5)
    leg = [(BLUE, "Multi-Head Attention"), (AMBER, "Feed-Forward Network"),
           (PURPLE, "残差连接 / 位置编码"), (GREEN, "LayerNorm"), (RED, "Mask（因果遮罩）")]
    lx = 48
    for col, txt in leg:
        s.rect(lx, 1546, 22, 14, fill=col, stroke="none", rx=3)
        s.text(lx + 30, 1559, txt, fs=12.5, fill=MUTED)
        lx += 248
    s.text(W - 48, 1559, "自绘：Python + Pillow；按“信息流 + 张量形状”重排，不是原图描红", fs=12, fill=MUTED, anchor="end")

    s.save("fig1-transformer-overview.png")
# =====================================================================
# 图 2：简化版数据流
# =====================================================================
def fig_simple():
    W, H = 1600, 880
    s = SVG(W, H, "#ffffff")
    s.text(40, 50, "简化版：一张图记住 Transformer 在做什么", fs=26, bold=True)
    s.text(40, 84, "信息流只有 4 个动作：加位置 → 注意力混合信息 → FFN 逐位置加工 → 残差+LN；Decoder 多一个“看原文”的交叉注意力",
           fs=13.5, fill=MUTED)
    s.line(40, 104, W - 40, 104, stroke="#e5e7eb", sw=1.5)

    row1 = [
        ("输入 token\nx1 … xn", "#ffffff", INK, ""),
        ("嵌入\n→ 512 维", BLUE_BG, BLUE, ""),
        ("+ 位置编码\n保住顺序信息", PURPLE_BG, PURPLE, ""),
        ("Self-Attention\n词与词混合信息", BLUE_BG, BLUE, ""),
    ]
    row2 = [
        ("Decoder 额外：\nCross-Attention（看原文）", TEAL_BG, TEAL, "只有解码器有"),
        ("FFN\n逐位置非线性", AMBER_BG, AMBER, ""),
        ("残差 + LayerNorm\n每步都做", GREEN_BG, GREEN, ""),
        ("Linear + Softmax\n输出下一个词", "#ffffff", INK, ""),
    ]
    bw, bh = 330, 96

    def draw_row(nodes, y, start_dir):
        x = 40
        for i, (label, bg, col, badge) in enumerate(nodes):
            s.rect(x, y, bw, bh, fill=bg, stroke=col, sw=2.2, rx=12)
            lines = label.split("\n")
            s.text(x + bw / 2, y + bh / 2 - (4 if len(lines) > 1 else -6), lines[0], fs=15,
                   anchor="middle", fill=col, bold=True)
            if len(lines) > 1:
                s.text(x + bw / 2, y + bh / 2 + 22, lines[1], fs=13, anchor="middle", fill=col)
            if badge:
                s.text(x + bw / 2, y - 8, badge, fs=11, anchor="middle", fill=col)
            if i < len(nodes) - 1:
                s.arrow_h(x + bw + 6, x + bw + 40, y + bh / 2, stroke=MUTED, sw=2)
            x += bw + 46

    s.rect(40, 124, W - 80, 34, fill="#f8fafc", stroke="#cbd5e1", sw=1.2, rx=8, dash="6 4")
    s.text(56, 147, "Encoder = 第 1 行整条链路 × 6 层（不含 Cross-Attention）", fs=13, fill=MUTED)
    s.text(W - 56, 147, "Decoder = 第 1、2 行整条链路 × 6 层", fs=13, fill=MUTED, anchor="end")

    draw_row(row1, 186, 1)
    # 折返箭头
    s.path("M 1490 250 C 1540 250 1540 268 1490 268", stroke=MUTED, sw=2)
    s.arrow_v(1490, 282, 340, stroke=MUTED, sw=2)
    draw_row(row2, 352, -1)
    s.text(1544, 318, "第 2 行从这里接着走 →", fs=11.5, fill=MUTED, anchor="end")

    notes = [
        (BLUE, "① 所有“搬运信息”的活都由注意力完成：可并行，任意两个位置的距离都是 1 步"),
        (AMBER, "② FFN 是唯一的非线性加工车间，但它只看单个位置，不负责混合词与词"),
        (PURPLE, "③ 没有位置编码，上面整张图对词序完全不敏感（“猫追狗” = “狗追猫”）"),
        (RED, "④ Decoder 训练时把整句并行喂入，靠 mask 假装“还没看到未来”"),
    ]
    y = 536
    for col, t in notes:
        s.rect(40, y - 13, 12, 12, fill=col, stroke="none", rx=2)
        s.text(62, y, t, fs=14, fill=INK)
        y += 40

    s.rect(40, 690, W - 80, 156, fill="#ffffff", stroke="#d1d5db", sw=1.4, rx=10)
    s.text(60, 720, "和 RNN / CNN 的最小对比（论文 Table 1 的直觉版）", fs=15, bold=True)
    s.text(60, 748, "RNN：第 t 步必须等第 t−1 步 → 不能并行；两个词相距 |i−j| 步 → 长依赖难学",
           fs=13, fill=MUTED)
    s.text(60, 772, "CNN：可以并行，但要让全句相连得堆 O(n/k) 层（膨胀卷积 O(log_k n) 层）",
           fs=13, fill=MUTED)
    s.text(60, 796, "Self-Attention：一步内任意两位置直连（路径长度 O(1)），代价是 O(n²·d) 的注意力矩阵",
           fs=13, fill=MUTED)
    s.text(60, 826, "⇒ 论文的取舍：用“更短的依赖路径 + 可并行”换掉“序列长度的平方复杂度”",
           fs=13.5, fill=INK)
    s.save("fig2-simplified-flow.png")


# =====================================================================
# 图 3：Attention 细节
# =====================================================================
def fig_attention():
    W, H = 1600, 900
    s = SVG(W, H, "#ffffff")
    s.text(40, 50, "Scaled Dot-Product Attention 拆解：一行 = 一个 query 的“提问—打分—汇总”", fs=24, bold=True)
    s.text(40, 84, "以 d_k = d_v = 4、3 个 query、5 个 key/value 为例；颜色与论文 Figure 2 的 Q / K / V 对应", fs=13.5, fill=MUTED)
    s.line(40, 104, W - 40, 104, stroke="#e5e7eb", sw=1.5)

    def mat(x, y, rows, cols, col, labels_r=None, labels_c=None, title="", sub="",
            cw=46, ch=38, vals=None, fs=11, pad=0):
        s.text(x, y - 44, title, fs=13.5, bold=True, fill=col)
        if sub:
            s.text(x, y - 24, sub, fs=11, fill=MUTED)
        for r in range(rows):
            for c in range(cols):
                s.rect(x + c * cw, y + r * ch, cw - 4, ch - 4, fill=col, fo=0.10,
                       stroke=col, sw=1, rx=3)
                if vals:
                    s.text(x + c * cw + (cw - 4) / 2, y + r * ch + ch / 2 + 4,
                           vals[r][c], fs=fs, fill=MUTED, anchor="middle")
        if labels_r:
            for r, t in enumerate(labels_r):
                s.text(x - 10 + pad, y + r * ch + ch / 2 + 4, t, fs=11.5, fill=col, anchor="end")
        if labels_c:
            for c, t in enumerate(labels_c):
                s.text(x + c * cw + (cw - 4) / 2, y - 6, t, fs=11, fill=MUTED, anchor="middle")

    mat(120, 250, 3, 4, BLUE, ["q1", "q2", "q3"], ["d1", "d2", "d3", "d4"],
        "Q（3 个 query）", "“我用什么去提问”")
    s.text(322, 320, "×", fs=30, fill=MUTED, anchor="middle")
    mat(400, 250, 4, 5, GREEN, ["d1", "d2", "d3", "d4"], ["k1", "k2", "k3", "k4", "k5"],
        "Kᵀ（5 个 key）", "已转置成 d_k × m")
    s.text(662, 320, "=", fs=30, fill=MUTED, anchor="middle")

    scores = [["3.2", "1.1", "0.4", "2.0", "0.2"],
              ["0.7", "2.8", "1.5", "0.9", "1.2"],
              ["1.4", "0.6", "3.1", "0.3", "2.2"]]
    mat(730, 250, 3, 5, PURPLE, ["q1", "q2", "q3"], ["k1", "k2", "k3", "k4", "k5"],
        "QKᵀ / √d_k（3 × 5）", "每一行 = 一个 query 对 5 个 key 的匹配分", cw=44, vals=scores)
    s.rect(728, 246, 216, 38, fill="none", stroke=PURPLE, sw=2.4, rx=4)

    s.arrow_h(966, 1030, 300, stroke=PURPLE, sw=2)
    s.rect(1040, 236, 190, 122, fill=PURPLE_BG, stroke=PURPLE, sw=2, rx=10)
    s.text(1135, 266, "softmax（逐行）", fs=14, bold=True, fill=PURPLE, anchor="middle")
    s.text(1135, 292, "0.46  0.10  0.04 …", fs=12, fill=INK, anchor="middle")
    s.text(1135, 314, "每行相加 = 1", fs=11.5, fill=MUTED, anchor="middle")
    s.text(1135, 336, "峰值越高 = 越聚焦", fs=11.5, fill=MUTED, anchor="middle")

    s.arrow_h(1236, 1300, 300, stroke=PURPLE, sw=2)
    mat(1320, 250, 3, 5, "#0e7490", ["q1", "q2", "q3"], ["k1", "k2", "k3", "k4", "k5"],
        "注意力权重 α", "行和 = 1，代表“分给每个词多少注意力”", cw=38, fs=9.5,
        vals=[[".46", ".10", ".04", ".25", ".15"],
              [".02", ".61", ".14", ".09", ".14"],
              [".05", ".01", ".72", ".01", ".21"]])

    # ---- 下半部分：加权求和 ----
    s.line(40, 424, W - 40, 424, stroke="#eef1f6", sw=1.5)
    s.text(48, 462, "再用 α 对 V 的每一行做加权求和：每个 query 得到一个新的 d_v 维向量（→ 进 Figure 2 的 Concat）",
           fs=14.5, bold=True, fill=INK)
    mat(120, 560, 5, 4, "#ca8a04", ["v1", "v2", "v3", "v4", "v5"], ["d1", "d2", "d3", "d4"],
        "V（5 个 value）", "“我实际提供的信息”", pad=28)
    s.text(330, 620, "×", fs=28, fill=MUTED, anchor="middle")
    mat(378, 560, 3, 5, "#0e7490", ["q1", "q2", "q3"], None, "α（3 × 5）", cw=38, pad=34)
    s.text(592, 620, "=", fs=28, fill=MUTED, anchor="middle")
    mat(628, 560, 3, 4, BLUE, ["o1", "o2", "o3"], ["d1", "d2", "d3", "d4"],
        "输出（3 × d_v）", "每个 query 一行", cw=50, ch=44)
    s.text(852, 596, "o1 = 0.46·v1 + 0.10·v2 + 0.04·v3 + 0.25·v4 + 0.15·v5", fs=12.5, fill=MUTED)
    s.text(852, 622, "o2、o3 同理：只换 α 的那一行", fs=12.5, fill=MUTED)
    s.text(852, 652, "注意 d_v 不必等于 d_k：Q、K 只用来算分数，", fs=12.5, fill=MUTED)
    s.text(852, 674, "V 才决定输出维度（论文里两者都取 64）", fs=12.5, fill=MUTED)

    s.rect(40, 760, W - 80, 104, fill="#f8fafc", stroke="#cbd5e1", sw=1.4, rx=8)
    s.text(60, 790, "为什么除以 √d_k：d_k 大时点积的方差随之变大，softmax 容易被推到“一口价”饱和区，梯度 ≈ 0；",
           fs=13, fill=INK)
    s.text(60, 814, "除以 √d_k 相当于把分数拉回单位方差附近，让 softmax 保持在“有梯度”的区间（论文脚注 4）。",
           fs=13, fill=INK)
    s.text(60, 840, "补一句：mask 就发生在 softmax 之前——把未来位置的分数设成 −∞，softmax 后自然变成 0。",
           fs=13, fill=RED)
    s.save("fig3-attention-detail.png")


if __name__ == "__main__":
    fig_overview()
    fig_simple()
    fig_attention()
