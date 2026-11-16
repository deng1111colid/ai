# -*- coding: utf-8 -*-
"""图 8：模型规模对比（3.18M / 8.77M / 38.5M）"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, C3, C4, GREY, GRID, OUT, plt, save  # noqa: E402

MODELS = [("tiny", 3.18, C3), ("main", 8.77, C1), ("big", 38.5, C4)]


def load(tag):
    return json.load(open(os.path.join(OUT, f"{tag}_metrics.json"), encoding="utf-8"))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))

    # (a) 测试集表现
    ax = axes[0]
    names = [f"{p}M 参数" for _, p, _ in MODELS]
    accs = [load(t)["test"]["acc"] * 100 for t, _, _ in MODELS]
    f1s = [load(t)["test"]["macro_f1"] * 100 for t, _, _ in MODELS]
    x = np.arange(len(MODELS))
    w = 0.34
    b1 = ax.bar(x - w / 2, accs, w, color=[c for _, _, c in MODELS], zorder=3,
                label="准确率")
    b2 = ax.bar(x + w / 2, f1s, w, color=[c for _, _, c in MODELS], alpha=0.45,
                zorder=3, hatch="//", edgecolor="white", label="宏平均 F1")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9,
                    f"{b.get_height():.2f}", ha="center", fontsize=9, color="#26313d")
    ax.set_xticks(x, names)
    ax.set_ylim(80, 96)
    ax.set_ylabel("测试集 (%)")
    ax.set_title("(a) 模型大 12 倍，准确率反而略低", fontsize=11, pad=8)
    ax.grid(axis="y", **GRID)

    # (b) 训练/验证损失曲线
    ax = axes[1]
    for tag, p, c in MODELS:
        m = load(tag)
        ep = [h["epoch"] for h in m["history"]]
        ax.plot(ep, [h["train_loss"] for h in m["history"]], "-", color=c, lw=1.8,
                marker="o", ms=4, label=f"{p}M 训练")
        ax.plot(ep, [h["val_loss"] for h in m["history"]], "--", color=c, lw=1.5,
                marker="s", ms=3.5, alpha=0.85, label=f"{p}M 验证")
    ax.set_xlabel("轮次")
    ax.set_ylabel("交叉熵损失")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_title("(b) 38.5M 模型训练损失最低，验证损失却最高", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper right")
    ax.grid(**GRID)

    save(fig, "f9_scale.png")

    for tag, p, _ in MODELS:
        m = load(tag)
        ts = [h["time"] for h in m["history"]]
        print(f"{tag:5s} 参数 {p:5.2f}M  acc {m['test']['acc']*100:5.2f}%  "
              f"宏F1 {m['test']['macro_f1']*100:5.2f}%  单轮 {np.median(ts):.1f}s  "
              f"末轮 train {m['history'][-1]['train_loss']:.3f} / "
              f"val {m['history'][-1]['val_loss']:.3f}")


if __name__ == "__main__":
    main()
