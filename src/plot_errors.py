# -*- coding: utf-8 -*-
"""图 10：错例分析（按"最自信却判错"排序）"""
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, GREY, OUT, plt, save  # noqa: E402


def wrap(s, n=22, lines=2):
    out = [s[i:i + n] for i in range(0, min(len(s), n * lines), n)]
    if len(s) > n * lines:
        out[-1] = out[-1][:-1] + "…"
    return "\n".join(out)


def main(tag="main"):
    a = json.load(open(os.path.join(OUT, f"analysis_{tag}.json"), encoding="utf-8"))
    errs = a["errors"]
    n = len(errs)

    fig, ax = plt.subplots(figsize=(12.6, 0.86 * n + 1.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 0.9)
    ax.axis("off")

    cols = [(0.12, "真实"), (0.95, "预测"), (1.75, "置信度"), (2.62, "评论内容")]
    for x, name in cols:
        ax.text(x, n + 0.42, name, fontsize=10, weight="bold", color="#3a4553")

    for i, e in enumerate(errs):
        y = n - i - 0.5
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.45), 10, 0.9, facecolor="#f6f8fb",
                                       edgecolor="none", zorder=1))
        true_name = "正面" if e["true"] == 1 else "负面"
        pred_name = "正面" if e["pred"] == 1 else "负面"
        ax.text(0.12, y, true_name, fontsize=10, color="#3a4553", va="center", zorder=2)
        ax.text(0.95, y, pred_name, fontsize=10, color=C2, va="center", zorder=2)
        ax.text(1.75, y, f"{e['prob']*100:.1f}%", fontsize=10, color=C2,
                va="center", zorder=2)
        ax.text(2.55, y, wrap(e["text"]), fontsize=9.4, color="#26313d",
                va="center", zorder=2, linespacing=1.45)

    n_miss = sum(1 for e in errs if e["true"] == 0 and e["pred"] == 1)
    ax.set_title(f"最自信的 {n} 个错误里有 {n_miss} 个是把差评读成了好评",
                 fontsize=12.5, weight="bold", pad=16)
    ax.text(0.12, 0.12, "注：「置信度」指模型判错的那个类别的概率", fontsize=8.8, color=GREY)
    fig.tight_layout()
    save(fig, "f6_errors.png")

    for e in errs[:4]:
        print(f"{e['prob']*100:5.1f}%  真实{e['true']} 预测{e['pred']}  {e['text'][:40]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
