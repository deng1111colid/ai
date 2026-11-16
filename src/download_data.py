# -*- coding: utf-8 -*-
"""下载数据集：酒店评论（主任务）与外卖评论（跨领域测试）"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

# GitHub 直连在国内不稳定，走 ghproxy 反代
FILES = {
    "ChnSentiCorp_htl_all.csv":
        "https://ghproxy.net/https://raw.githubusercontent.com/SophonPlus/"
        "ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
    "waimai_10k.csv":
        "https://ghproxy.net/https://raw.githubusercontent.com/SophonPlus/"
        "ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
}


def main():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            print(f"{name} 已存在，跳过")
            continue
        print(f"下载 {name} ...")
        subprocess.run(["curl", "-sL", "--max-time", "180", "-o", dst, url], check=True)
        print(f"  {os.path.getsize(dst)/2**20:.2f} MB")


if __name__ == "__main__":
    main()
