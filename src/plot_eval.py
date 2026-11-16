# -*- coding: utf-8 -*-
"""图 3 混淆矩阵、图 4 置信度与校准、图 5 长度分层准确率"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, C3, C4, GREY, GRID, OUT, plt, save  # noqa: E402


def fig_confusion(a):
    t = a["test"]
    c = t["confusion"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    ax = axes[0]
    im = ax.imshow(mat, cmap="Blues", alpha=0.9)
    for i in range(2):
        for j in range(2):
            v = int(mat[i, j])
            pct = mat[i, j] / mat[i].sum() * 100
            color = "white" if mat[i, j] > mat.max() * 0.55 else "#26313d"
            ax.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=13, color=color, linespacing=1.5)
    ax.set_xticks([0, 1], ["预测为负面", "预测为正面"])
    ax.set_yticks([0, 1], ["真实负面", "真实正面"])
    ax.set_title(f"(a) 混淆矩阵：整体准确率 {t['acc']*100:.2f}%", fontsize=11, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)

    ax = axes[1]
    labels = ["负面", "正面"]
    prec = [t["prec_neg"], t["prec_pos"]]
    rec = [t["rec_neg"], t["rec_pos"]]
    f1 = [t["f1_neg"], t["f1_pos"]]
    x = np.arange(2)
    w = 0.26
    for k, (vals, col, name) in enumerate([(prec, C1, "精确率"), (rec, C3, "召回率"),
                                           (f1, C4, "F1")]):
        bars = ax.bar(x + (k - 1) * w, [v * 100 for v in vals], w, color=col,
                      label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.2, f"{v*100:.1f}",
                    ha="center", fontsize=8.6, color="#5b6472")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("百分比 (%)")
    ax.set_title(f"(b) 分类指标：宏平均 F1 {t['macro_f1']*100:.2f}%", fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="y", **GRID)

    save(fig, "f3_confusion.png")


def fig_confidence(a):
    t = a["test"]
    h = a["conf_hist"]
    edges = np.array(h["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    correct = np.array(h["correct"])
    wrong = np.array(h["wrong"])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))

    ax = axes[0]
    ax.bar(centers, correct, width=0.042, color=C3, label="预测正确", zorder=3)
    ax.bar(centers, wrong, width=0.042, bottom=correct, color=C2, label="预测错误", zorder=3)
    ax.set_xlabel("模型给出的置信度（取较大的一类概率）")
    ax.set_ylabel("样本数")
    ax.set_xlim(0.48, 1.02)
    ax.set_title("(a) 绝大多数样本落在 0.9 以上", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", **GRID)

    ax = axes[1]
    rel = [r for r in a["reliability"] if r["n"] >= 10]   # 样本太少的箱不画，避免折线乱跳
    xs = [r["conf"] for r in rel]
    ys = [r["acc"] for r in rel]
    ns = [r["n"] for r in rel]
    ax.plot([0.5, 1.0], [0.5, 1.0], ls="--", color=GREY, lw=1.2, label="完美校准")
    sc = ax.scatter(xs, ys, s=[max(30, n / 2.2) for n in ns], color=C1, zorder=4,
                    edgecolor="white", lw=1.2)
    ax.plot(xs, ys, color=C1, lw=1.2, alpha=0.55, zorder=3)
    ax.set_xlim(0.52, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("平均置信度")
    ax.set_ylabel("真实准确率")
    ax.set_title(f"(b) 校准曲线：ECE = {t['ece']*100:.2f}%，点的大小代表样本数",
                 fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(**GRID)

    save(fig, "f4_confidence.png")


def fig_length(a):
    bs = a["length_buckets"]
    labels = [b["bucket"] for b in bs]
    accs = [b["acc"] * 100 for b in bs]
    ns = [b["n"] for b in bs]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bars = ax.bar(range(len(bs)), accs, color=C1, width=0.58, zorder=3)
    ax.axhline(a["test"]["acc"] * 100, color=C2, ls="--", lw=1.4, zorder=4,
               label=f"整体准确率 {a['test']['acc']*100:.2f}%")
    for i, (b, v, n) in enumerate(zip(bars, accs, ns)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                ha="center", fontsize=10, color="#26313d")
        ax.text(b.get_x() + b.get_width() / 2, 4, f"n={n}", ha="center",
                fontsize=9, color="white")
    ax.set_xticks(range(len(bs)), labels)
    ax.set_xlabel("评论字数")
    ax.set_ylabel("准确率 (%)")
    ax.set_ylim(0, 112)
    ax.set_title("按评论长度分层的准确率：200 字以上的长评论明显更差", fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", **GRID)
    save(fig, "f5_length.png")


def main(tag="main"):
    a = json.load(open(os.path.join(OUT, f"analysis_{tag}.json"), encoding="utf-8"))
    fig_confusion(a)
    fig_confidence(a)
    fig_length(a)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
