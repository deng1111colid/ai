# -*- coding: utf-8 -*-
"""清洗语料、划分数据集、输出统计

输入: data/raw/ChnSentiCorp_htl_all.csv  酒店评论(主任务)
      data/raw/waimai_10k.csv           外卖评论(跨领域测试)
输出: data/train.tsv data/val.tsv data/test.tsv data/ood.tsv data/stats.json
"""
import csv
import json
import os
import random
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data")

SEED = 42
VAL_RATIO, TEST_RATIO = 0.1, 0.1
MAX_LEN = 256


def read_csv(name):
    """读 label,review 两列，返回 [(text, label)]"""
    rows = []
    with open(os.path.join(RAW, name), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            text = (r["review"] or "").strip().replace("\t", " ").replace("\n", " ")
            if len(text) < 2:
                continue
            rows.append((text, int(r["label"])))
    return rows


def dedup(rows):
    seen, out = set(), []
    for t, y in rows:
        if t in seen:
            continue
        seen.add(t)
        out.append((t, y))
    return out


def split(rows, seed=SEED):
    """按标签分层抽样，保证训练/验证/测试的正负比例一致"""
    rng = random.Random(seed)
    by_label = {}
    for t, y in rows:
        by_label.setdefault(y, []).append((t, y))
    train, val, test = [], [], []
    for y, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_test, n_val = int(n * TEST_RATIO), int(n * VAL_RATIO)
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def length_stat(texts):
    L = sorted(len(t) for t in texts)
    q = lambda p: L[min(len(L) - 1, int(len(L) * p))]
    return {
        "min": L[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
        "p90": q(0.90), "p99": q(0.99), "max": L[-1],
        "mean": round(sum(L) / len(L), 1),
        "over_max_len": sum(1 for x in L if x > MAX_LEN),
        "over_max_len_ratio": round(sum(1 for x in L if x > MAX_LEN) / len(L), 4),
    }


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["label", "text"])
        for t, y in rows:
            w.writerow([y, t])


def main():
    htl = dedup(read_csv("ChnSentiCorp_htl_all.csv"))
    ood = dedup(read_csv("waimai_10k.csv"))
    print(f"[1/4] 酒店评论 {len(htl)} 条，外卖评论 {len(ood)} 条")

    train, val, test = split(htl)
    print(f"[2/4] 训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}")

    write_tsv(os.path.join(OUT, "train.tsv"), train)
    write_tsv(os.path.join(OUT, "val.tsv"), val)
    write_tsv(os.path.join(OUT, "test.tsv"), test)
    write_tsv(os.path.join(OUT, "ood.tsv"), ood)

    stats = {
        "seed": SEED,
        "max_len": MAX_LEN,
        "splits": {
            "train": len(train), "val": len(val), "test": len(test), "total": len(htl),
        },
        "label_dist": {
            "all": dict(Counter(y for _, y in htl)),
            "train": dict(Counter(y for _, y in train)),
            "val": dict(Counter(y for _, y in val)),
            "test": dict(Counter(y for _, y in test)),
        },
        "length": length_stat([t for t, _ in htl]),
        "length_by_label": {
            str(y): length_stat([t for t, yy in htl if yy == y]) for y in (0, 1)
        },
        "ood": {
            "n": len(ood),
            "label_dist": dict(Counter(y for _, y in ood)),
            "length": length_stat([t for t, _ in ood]),
        },
        "examples": {
            "pos": [t for t, y in htl if y == 1][:5],
            "neg": [t for t, y in htl if y == 0][:5],
        },
    }
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[3/4] 长度中位数 {stats['length']['median']}，超过 {MAX_LEN} 字的占 "
          f"{stats['length']['over_max_len_ratio']*100:.1f}%")
    print(f"[4/4] 正负比 训练集 {stats['label_dist']['train']}")
    print("预览:", train[0][1], train[0][0][:40])


if __name__ == "__main__":
    main()
