# -*- coding: utf-8 -*-
"""图 1：数据概况（标签分布、长度分布、正负样本长度差异）"""
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, C3, GREY, GRID, ROOT, plt, save  # noqa: E402


def read_split(name):
    rows = []
    with open(os.path.join(ROOT, "data", f"{name}.tsv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["text"], int(r["label"])))
    return rows


def main():
    rows = read_split("train") + read_split("val") + read_split("test")
    stats = json.load(open(os.path.join(ROOT, "data", "stats.json"), encoding="utf-8"))
    lens = np.array([len(t) for t, _ in rows])
    y = np.array([l for _, l in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))

    # (a) 标签分布
    ax = axes[0]
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    bars = ax.bar(["正面", "负面"], [pos, neg], color=[C1, C2], width=0.55, zorder=3)
    total = pos + neg
    for b, v in zip(bars, [pos, neg]):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.015,
                f"{v:,}\n（{v/total*100:.1f}%）", ha="center", fontsize=10, color="#3a4553")
    ax.set_ylim(0, pos * 1.28)
    ax.set_ylabel("评论条数")
    ax.set_title("(a) 标签分布：正面评论占多数", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)

    # (b) 长度分布
    ax = axes[1]
    ax.hist(lens, bins=np.arange(0, 520, 12), color=C1, alpha=0.85, zorder=3)
    med = int(np.median(lens))
    ax.axvline(med, color=C3, ls="--", lw=1.4, zorder=4)
    ax.text(med + 8, ax.get_ylim()[1] * 0.88, f"中位数 {med} 字", color=C3, fontsize=9.5)
    ax.axvline(256, color=C2, ls=":", lw=1.4, zorder=4)
    ax.text(262, ax.get_ylim()[1] * 0.62, "截断长度 256", color=C2, fontsize=9.5)
    ax.set_xlabel("评论字数")
    ax.set_ylabel("条数")
    ax.set_title("(b) 长度分布：长尾明显，11% 超过 256 字", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)

    # (c) 正负样本长度对比
    ax = axes[2]
    bins = np.arange(0, 400, 15)
    ax.hist([lens[y == 1], lens[y == 0]], bins=bins, density=True,
            color=[C1, C2], alpha=0.72, label=["正面", "负面"], zorder=3)
    ax.set_xlabel("评论字数")
    ax.set_ylabel("密度")
    ax.set_title("(c) 两类长度分布几乎重合", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(axis="y", **GRID)

    fig.suptitle(f"酒店评论情感数据集：{total:,} 条", fontsize=13, weight="bold", y=1.04)
    save(fig, "f1_data.png")
    print("长度分位", stats["length"])


if __name__ == "__main__":
    main()
