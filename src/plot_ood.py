# -*- coding: utf-8 -*-
"""图 6：跨领域泛化（酒店评论 → 外卖评论）"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import C1, C2, C3, C4, GREY, GRID, OUT, plt, save  # noqa: E402


def main(tag="main"):
    a = json.load(open(os.path.join(OUT, f"analysis_{tag}.json"), encoding="utf-8"))
    mt, mo = a["test"], a["ood"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))

    # (a) 三项指标对比
    ax = axes[0]
    names = ["准确率", "宏平均 F1", "AUC"]
    v_in = [mt["acc"] * 100, mt["macro_f1"] * 100, mt["auc"] * 100]
    v_out = [mo["acc"] * 100, mo["macro_f1"] * 100, mo["auc"] * 100]
    x = np.arange(3)
    w = 0.34
    b1 = ax.bar(x - w / 2, v_in, w, color=C1, label="域内（酒店评论）", zorder=3)
    b2 = ax.bar(x + w / 2, v_out, w, color=C4, label="跨域（外卖评论）", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.3,
                    f"{b.get_height():.1f}", ha="center", fontsize=9.5, color="#26313d")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 112)
    ax.set_ylabel("百分比 (%)")
    ax.set_title("(a) 准确率掉 20 个点，AUC 只掉 8 个点", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9.2, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.09))
    ax.grid(axis="y", **GRID)

    # (b) 两个域的正类概率累计分布
    ax = axes[1]
    for key, col, name, m in [("test", C1, "域内（酒店）", mt), ("ood", C4, "跨域（外卖）", mo)]:
        c = a["prob_cdf"][key]
        ax.plot(c["x"], c["y"], color=col, lw=2.0, label=f"{name}：真实正类 "
                                                        f"{m['pos_rate_true']*100:.0f}%")
        hit = next((y for x, y in zip(c["x"], c["y"]) if x >= 0.5), 1.0)
        ax.scatter([0.5], [hit], s=70, color=col, zorder=5, edgecolor="white", lw=1.2)
        ax.annotate(f"阈值 0.5 以上占 {1-hit:.0%}", (0.5, hit),
                    textcoords="offset points",
                    xytext=(14, -4) if key == "test" else (14, 10),
                    fontsize=9, color=col, ha="left",
                    va="top" if key == "test" else "bottom")
    ax.axvline(0.5, color=GREY, ls="--", lw=1.1)
    ax.set_xlabel("模型预测为「正面」的概率")
    ax.set_ylabel("累计样本占比")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) 同一阈值下，跨域被判为正面的比例明显更多", fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(**GRID)

    fig.suptitle("同一模型换一个领域：排序能力保住了，判定阈值失效了",
                 fontsize=12.5, weight="bold", y=1.03)
    save(fig, "f7_ood.png")
    print(f"域内预测正类占比 {mt['pos_rate_pred']*100:.1f}% / 跨域 "
          f"{mo['pos_rate_pred']*100:.1f}% / 真实 {mo['pos_rate_true']*100:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
