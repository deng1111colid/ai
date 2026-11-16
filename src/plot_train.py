# -*- coding: utf-8 -*-
"""图 2：训练过程（损失、验证指标、学习率与梯度范数）"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, C3, C4, GREY, GRID, OUT, plt, save  # noqa: E402


def smooth(x, w=20):
    if len(x) < w:
        return np.array(x)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main(tag="main"):
    m = json.load(open(os.path.join(OUT, f"{tag}_metrics.json"), encoding="utf-8"))
    h, st = m["history"], m["steps"]
    ep = [x["epoch"] for x in h]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2))

    # (a) 损失
    ax = axes[0]
    steps = [s["step"] for s in st]
    loss = [s["loss"] for s in st]
    ax.plot(steps, loss, color=C1, alpha=0.16, lw=0.9, label="单步训练损失")
    sm = smooth(loss, 20)
    ax.plot(range(10, 10 + len(sm)), sm, color=C1, lw=2.0, label="平滑后")
    # 每轮末尾的验证损失，横坐标换算到对应步数
    per = len(st) // len(ep)
    ax.plot([e * per for e in ep], [x["val_loss"] for x in h], "s-", color=C2,
            ms=6, lw=1.6, label="每轮验证损失")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("(a) 训练损失与验证损失同步下降", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(**GRID)

    # (b) 验证指标
    ax = axes[1]
    ax.plot(ep, [x["val_acc"] * 100 for x in h], "o-", color=C1, lw=1.8,
            ms=5, label="验证准确率")
    ax.plot(ep, [x["val_macro_f1"] * 100 for x in h], "s-", color=C3, lw=1.8,
            ms=5, label="验证宏平均 F1")
    best = max(h, key=lambda x: x["val_macro_f1"])
    ax.axvline(best["epoch"], color=C4, ls="--", lw=1.3, alpha=0.75,
               label=f"最佳轮次（第 {best['epoch']} 轮）")
    ax.axhline(m["test"]["acc"] * 100, color=GREY, ls=":", lw=1.3,
               label=f"测试集准确率 {m['test']['acc']*100:.2f}%")
    ax.set_xlabel("轮次")
    ax.set_ylabel("百分比 (%)")
    ax.set_ylim(70, 95)
    ax.set_xticks(ep)
    ax.set_title("(b) 验证指标在第 3 轮后趋于平稳", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")
    ax.grid(**GRID)

    # (c) 学习率与梯度范数
    ax = axes[2]
    ax.plot(steps, [s["lr"] for s in st], color=C2, lw=1.6, label="学习率")
    ax.set_xlabel("训练步数")
    ax.set_ylabel(r"学习率 ($10^{-5}$)", color=C2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*1e5:.1f}"))
    ax.tick_params(axis="y", colors=C2)
    ax.grid(**GRID)
    ax2 = ax.twinx()
    ax2.plot(steps, [s["grad_norm"] for s in st], color=C1, alpha=0.45, lw=0.9)
    ax2.set_ylabel("梯度范数", color=C1)
    ax2.tick_params(axis="y", colors=C1)
    ax2.spines["right"].set_color(C1)
    ax.set_title("(c) 线性预热 + 线性衰减，梯度始终平稳", fontsize=11, pad=8)

    fig.suptitle(f"{m['config']['model']}　{m['params']:,} 参数　训练 {m['total_time']} 秒",
                 fontsize=12.5, weight="bold", y=1.03)
    save(fig, "f2_train.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
