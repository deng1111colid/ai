# -*- coding: utf-8 -*-
"""图 7：消融实验（训练数据量、学习率）"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, C3, C4, GREY, GRID, OUT, plt, save  # noqa: E402


def load(tag):
    return json.load(open(os.path.join(OUT, f"{tag}_metrics.json"), encoding="utf-8"))


def main():
    fracs = [("frac10", 0.1), ("frac25", 0.25), ("frac50", 0.5), ("main", 1.0)]
    xs, accs, f1s, ns = [], [], [], []
    for tag, _ in fracs:
        m = load(tag)
        xs.append(m["n_train"])
        accs.append(m["test"]["acc"] * 100)
        f1s.append(m["test"]["macro_f1"] * 100)
        ns.append(m["n_train"])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3))

    # (a) 数据量
    ax = axes[0]
    ax.plot(xs, accs, "o-", color=C1, lw=2.0, ms=7, label="准确率")
    ax.plot(xs, f1s, "s-", color=C3, lw=2.0, ms=6.5, label="宏平均 F1")
    for x, a, f in zip(xs, accs, f1s):
        ax.annotate(f"{a:.2f}", (x, a), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=C1)
        ax.annotate(f"{f:.2f}", (x, f), textcoords="offset points", xytext=(0, -17),
                    ha="center", fontsize=9, color=C3)
    ax.text(0.03, 0.06, f"数据量 ×10：准确率 +{accs[-1]-accs[0]:.1f} 点，"
                        f"宏平均 F1 +{f1s[-1]-f1s[0]:.1f} 点",
            transform=ax.transAxes, fontsize=9.0, color=GREY, va="bottom")
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("测试集表现 (%)")
    ax.set_ylim(74, 96)
    ax.set_xlim(200, 6900)
    ax.set_title("(a) 数据量消融：前 25% 就拿走了大部分收益", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9.2, loc="lower right")
    ax.grid(**GRID)

    # (b) 学习率（都在 3.18M 的小模型上跑）
    ax = axes[1]
    lrs = [("lr1e5", 1e-5), ("tiny", 5e-5), ("lr1e4", 1e-4),
           ("lr2e4", 2e-4), ("lr25e4", 2.5e-4), ("lr5e4", 5e-4)]
    lx, la, lf = [], [], []
    for tag, lr in lrs:
        m = load(tag)
        lx.append(lr)
        la.append(m["test"]["acc"] * 100)
        lf.append(m["test"]["macro_f1"] * 100)
    ax.semilogx(lx, la, "o-", color=C1, lw=2.0, ms=7, label="准确率")
    ax.semilogx(lx, lf, "s-", color=C3, lw=2.0, ms=6.5, label="宏平均 F1")
    for x, a, f in zip(lx, la, lf):
        ax.annotate(f"{a:.1f}", (x, a), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=C1)
        ax.annotate(f"{f:.1f}", (x, f), textcoords="offset points", xytext=(0, -17),
                    ha="center", fontsize=9, color=C3)
    ax.set_xlim(lx[0] / 3, lx[-1] * 4)
    ax.set_xticks(lx, ["1e-5", "5e-5", "1e-4", "2e-4", "2.5e-4", "5e-4"],
                  fontsize=8.6, rotation=28, ha="right")
    ax.set_xlabel("峰值学习率")
    ax.set_ylabel("测试集表现 (%)")
    ax.set_ylim(min(lf) - 6, max(la) + 8)
    ax.set_title("(b) 学习率：过小欠拟合，过大过拟合", fontsize=11, pad=8)
    ax.axvspan(lx[0] / 3, lx[0] * 1.4, color=C2, alpha=0.07, zorder=1)
    ax.axvspan(lx[-1] / 1.4, lx[-1] * 4, color=C2, alpha=0.07, zorder=1)
    ax.legend(frameon=False, fontsize=9.2, loc="lower right")
    ax.grid(**GRID)

    save(fig, "f8_ablation.png")
    print("数据量", list(zip(ns, [round(a, 2) for a in accs])))
    print("学习率", list(zip(lx, [round(a, 2) for a in la])))


if __name__ == "__main__":
    main()
