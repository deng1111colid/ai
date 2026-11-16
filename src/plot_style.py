# -*- coding: utf-8 -*-
"""所有绘图脚本共用的样式"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "outputs")

for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
          "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c8ced6",
    "axes.labelcolor": "#3a4553",
    "text.color": "#26313d",
    "xtick.color": "#4a5563",
    "ytick.color": "#4a5563",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "savefig.dpi": 170,
})

C1 = "#2f6f9f"     # 主色（蓝）
C2 = "#c0504d"     # 强调 / 错误（红）
C3 = "#4f8a5b"     # 正确 / 正向（绿）
C4 = "#8a6fbf"     # 第四色（紫）
GREY = "#7b8794"
GRID = dict(color="#e6eaee", lw=0.8, zorder=0)


def save(fig, name):
    p = os.path.join(ASSETS, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)
