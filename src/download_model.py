# -*- coding: utf-8 -*-
"""从 hf-mirror 镜像下载预训练模型（HuggingFace 官方站国内不可达）

镜像会把文件 302 到 /api/resolve-cache/，Python 的 urllib 处理这个跳转不稳，
所以直接用 curl。
"""
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

MIRROR = "https://hf-mirror.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 只取 PyTorch 训练需要的文件，跳过 flax/tf 权重
FILES = ["config.json", "vocab.txt", "tokenizer_config.json",
         "special_tokens_map.json", "pytorch_model.bin"]

MODELS = [
    "uer/chinese_roberta_L-4_H-256",   # 主模型，8.77M 参数
    "uer/chinese_roberta_L-2_H-128",   # 最小档，3.18M
    "hfl/rbt3",                        # 最大档，38.5M
]


def fetch(repo, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in FILES:
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = f"{MIRROR}/{repo}/resolve/main/{name}"
        r = subprocess.run(["curl", "-sL", "--max-time", "300", "-o", dst, url])
        if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            if os.path.exists(dst):
                os.remove(dst)
            print(f"  {name:24s} 下载失败")
            continue
        print(f"  {name:24s} {os.path.getsize(dst)/2**20:7.1f} MB")


def main():
    models = sys.argv[1:] or MODELS
    for repo in models:
        tag = repo.split("/")[-1]
        print(f"[{repo}]")
        fetch(repo, os.path.join(ROOT, "models", tag))


if __name__ == "__main__":
    main()
