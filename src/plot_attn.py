# -*- coding: utf-8 -*-
"""图 9：注意力热图（最后一层 [CLS] 对各 token 的注意力）"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, C3, GREY, GRID, OUT, plt, save  # noqa: E402


def main(tag="main"):
    a = json.load(open(os.path.join(OUT, f"analysis_{tag}.json"), encoding="utf-8"))
    cases = a["attention"]

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 7.6))
    for ax, c in zip(axes.ravel(), cases):
        toks, ws = c["tokens"], c["weights"]
        # 去掉特殊符号，取权重最高的 10 个 token
        pairs = [(t, w) for t, w in zip(toks, ws) if t not in ("[CLS]", "[SEP]", "[UNK]")]
        pairs = sorted(pairs, key=lambda x: -x[1])[:10][::-1]
        labels = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(vals)))
        bars = ax.barh(range(len(vals)), vals, color=colors, height=0.68, zorder=3)
        ax.set_yticks(range(len(vals)), labels, fontsize=12)
        ax.set_xlabel("注意力权重（归一化后）")
        m = max(vals)
        for b, v in zip(bars, vals):
            ax.text(v + m * 0.02, b.get_y() + b.get_height() / 2, f"{v*100:.1f}%",
                    va="center", fontsize=8, color="#5b6472")
        ax.set_xlim(0, m * 1.2)
        ax.grid(axis="x", **GRID)
        head = c["text"][:22] + ("…" if len(c["text"]) > 22 else "")
        ax.set_title(f"「{head}」　预测正类概率 {c['prob']*100:.1f}%",
                     fontsize=10, pad=8)

    fig.suptitle("模型在关注哪些字：最后一层 [CLS] 位置的注意力（按头平均，取前 10）",
                 fontsize=12.5, weight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "f10_attention.png")

    for c in cases:
        pairs = sorted([(t, w) for t, w in zip(c["tokens"], c["weights"])
                        if t not in ("[CLS]", "[SEP]")], key=lambda x: -x[1])[:6]
        print(f"{c['text'][:26]:28s} -> " + " ".join(f"{t}{w*100:.0f}%" for t, w in pairs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
