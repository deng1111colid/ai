# -*- coding: utf-8 -*-
"""对训练好的模型做完整评估：测试集 / 跨领域 / 置信度校准 / 长度分层 / 注意力

用法: python src/analyze.py --tag main
输出: outputs/analysis_{tag}.json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import collate, metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


def read_tsv(split):
    rows = []
    with open(os.path.join(ROOT, "data", f"{split}.tsv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["text"], int(r["label"])))
    return rows


class Cls(Dataset):
    def __init__(self, rows, tok, max_len):
        enc = tok([t for t, _ in rows], truncation=True, max_length=max_len)
        self.ids = enc["input_ids"]
        self.labels = [y for _, y in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def auc_score(y, p):
    """Mann-Whitney 形式的 AUC，不依赖 sklearn"""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability(y, p, n_bins=10):
    """置信度校准：把样本按最大概率分箱，比较平均置信度与真实准确率"""
    y = np.asarray(y)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    bins, edges = [], np.linspace(0.5, 1.0, n_bins + 1)
    ece, total = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        bins.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                     "conf": float(conf[m].mean()), "acc": float(correct[m].mean())})
        ece += abs(conf[m].mean() - correct[m].mean()) * m.sum() / total
    return bins, float(ece)


def length_bucket(n):
    if n < 50:
        return "<50"
    if n < 100:
        return "50-100"
    if n < 150:
        return "100-150"
    if n < 200:
        return "150-200"
    return "200+"


def run(model, tok, rows, device, max_len):
    ds = Cls(rows, tok, max_len)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for ids, mask, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            probs.append(torch.softmax(out.logits.float(), -1)[:, 1].cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def bucket_stats(rows, p, y):
    """按评论长度分组统计准确率"""
    groups = {}
    pred = (p >= 0.5).astype(int)
    for (text, _), pi, yi in zip(rows, pred, y):
        groups.setdefault(length_bucket(len(text)), []).append(int(pi == yi))
    order = ["<50", "50-100", "100-150", "150-200", "200+"]
    out = []
    for k in order:
        if k in groups:
            v = groups[k]
            out.append({"bucket": k, "n": len(v), "acc": sum(v) / len(v)})
    return out


@torch.no_grad()
def attention_of(model, tok, cases, device, max_len):
    """取最后一层 [CLS] 位置对各个 token 的注意力，按头平均"""
    out = []
    for text in cases:
        enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        res = model(input_ids=ids, attention_mask=mask, output_attentions=True)
        att = res.attentions[-1][0].float().mean(0)[0]          # (T,)
        toks = tok.convert_ids_to_tokens(ids[0].tolist())
        keep = [(t, float(w)) for t, w in zip(toks, att.tolist())
                if t not in ("[PAD]",)]
        s = sum(w for _, w in keep) or 1.0
        prob = float(torch.softmax(res.logits.float(), -1)[0, 1])
        out.append({"text": text, "tokens": [t for t, _ in keep],
                    "weights": [w / s for _, w in keep], "prob": prob})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n_attn", type=int, default=4)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(OUT, f"{args.tag}_metrics.json"), encoding="utf-8"))
    name = meta["config"]["model"]
    max_len = meta["config"]["max_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = BertTokenizer.from_pretrained(os.path.join(ROOT, "models", name))
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(ROOT, "models", name), num_labels=2).to(device)
    model.load_state_dict(torch.load(os.path.join(OUT, f"{args.tag}_ckpt.pt"), map_location=device))

    test_rows, ood_rows = read_tsv("test"), read_tsv("ood")
    p_test, y_test = run(model, tok, test_rows, device, max_len)
    p_ood, y_ood = run(model, tok, ood_rows, device, max_len)

    mt = metrics(y_test, p_test)
    mt["auc"] = auc_score(y_test, p_test)
    mo = metrics(y_ood, p_ood)
    mo["auc"] = auc_score(y_ood, p_ood)
    mo["pos_rate_pred"] = float((p_ood >= 0.5).mean())
    for m in (mt, mo):
        pred = (np.asarray(p_test if m is mt else p_ood) >= 0.5).astype(int)
        yy = y_test if m is mt else y_ood
        m["pos_rate_true"] = float((np.asarray(yy) == 1).mean())
    pred_t = (p_test >= 0.5).astype(int)
    mt["pos_rate_pred"] = float(pred_t.mean())

    def prob_hist(p, bins=20):
        c, e = np.histogram(p, bins=np.linspace(0, 1, bins + 1))
        return {"edges": e.tolist(), "counts": c.tolist()}

    def cdf(p, n=400):
        s = np.sort(p)
        m = len(s)
        idx = np.linspace(0, m - 1, min(n, m)).astype(int)
        return {"x": s[idx].tolist(), "y": ((idx + 1) / m).tolist()}

    rel, ece = reliability(y_test, p_test)
    mt["ece"] = ece

    # 置信度直方图数据（正确 / 错误分开）
    conf = np.maximum(p_test, 1 - p_test)
    correct = pred_t == y_test
    hist_edges = np.linspace(0.5, 1.0, 11)
    hist = {"edges": hist_edges.tolist(), "correct": [], "wrong": []}
    for i in range(10):
        m = (conf >= hist_edges[i]) & (conf < hist_edges[i + 1]) if i < 9 else \
            (conf >= hist_edges[i]) & (conf <= hist_edges[i + 1])
        hist["correct"].append(int((m & correct).sum()))
        hist["wrong"].append(int((m & ~correct).sum()))

    # 错例：按"自信但错"排序
    err_idx = np.where(~correct)[0]
    err_idx = err_idx[np.argsort(-conf[err_idx])][:8]
    errors = [{
        "text": test_rows[i][0][:120],
        "true": int(y_test[i]), "pred": int(pred_t[i]),
        "prob": float(p_test[i]), "len": len(test_rows[i][0]),
    } for i in err_idx]

    # 注意力示例：挑两正两负，取中等长度
    cand = [i for i in range(len(test_rows)) if 30 <= len(test_rows[i][0]) <= 80]
    pos = [i for i in cand if y_test[i] == 1][: args.n_attn // 2]
    neg = [i for i in cand if y_test[i] == 0][: args.n_attn // 2]
    cases = [test_rows[i][0] for i in pos + neg]
    attn = attention_of(model, tok, cases, device, max_len)

    result = {
        "tag": args.tag, "model": name, "params": meta["params"],
        "test": mt, "ood": mo,
        "reliability": rel, "conf_hist": hist,
        "length_buckets": bucket_stats(test_rows, p_test, y_test),
        "errors": errors, "attention": attn,
        "prob_hist": {"test": prob_hist(p_test), "ood": prob_hist(p_ood)},
        "prob_cdf": {"test": cdf(p_test), "ood": cdf(p_ood)},
        "best_epoch": meta["best_epoch"], "total_time": meta["total_time"],
        "n_steps": meta["n_steps"],
    }
    with open(os.path.join(OUT, f"analysis_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"{args.tag}: test acc {mt['acc']*100:.2f}%  macroF1 {mt['macro_f1']*100:.2f}%  "
          f"AUC {mt['auc']:.4f}  ECE {ece:.4f}  | ood acc {mo['acc']*100:.2f}%  "
          f"AUC {mo['auc']:.4f}")
    print("长度分层:", [(b['bucket'], round(b['acc'] * 100, 1), b['n']) for b in
                    result["length_buckets"]])


if __name__ == "__main__":
    main()
