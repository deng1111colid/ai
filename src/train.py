# -*- coding: utf-8 -*-
"""在酒店评论上微调小模型做情感二分类

用法:
    python src/train.py --model chinese_roberta_L-4_H-256 --tag main
    python src/train.py --model chinese_roberta_L-2_H-128 --tag tiny
    python src/train.py --frac 0.25 --tag frac25
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (BertForSequenceClassification, BertTokenizer,
                          get_linear_schedule_with_warmup)

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
MODELS = os.path.join(ROOT, "models")


def read_tsv(split):
    rows = []
    with open(os.path.join(DATA, f"{split}.tsv"), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append((r["text"], int(r["label"])))
    return rows


class Cls(Dataset):
    def __init__(self, rows, tok, max_len):
        texts = [t for t, _ in rows]
        enc = tok(texts, truncation=True, max_length=max_len, add_special_tokens=True)
        self.ids = enc["input_ids"]
        self.labels = [y for _, y in rows]
        self.lens = [len(x) for x in self.ids]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    maxlen = max(len(x[0]) for x in batch)
    ids = torch.zeros(len(batch), maxlen, dtype=torch.long)
    mask = torch.zeros(len(batch), maxlen, dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
    return ids, mask, labels


def metrics(y, p, thr=0.5):
    """y 真实标签, p 预测为正类的概率"""
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    prec1 = tp / (tp + fp) if tp + fp else 0.0
    rec1 = tp / (tp + fn) if tp + fn else 0.0
    f1_1 = 2 * prec1 * rec1 / (prec1 + rec1) if prec1 + rec1 else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    rec0 = tn / (tn + fp) if tn + fp else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
    return {
        "acc": acc, "macro_f1": (f1_0 + f1_1) / 2,
        "f1_pos": f1_1, "f1_neg": f1_0,
        "prec_pos": prec1, "rec_pos": rec1,
        "prec_neg": prec0, "rec_neg": rec0,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate(model, loader, device, amp):
    model.eval()
    loss_sum, n = 0.0, 0
    probs, labels = [], []
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(input_ids=ids, attention_mask=mask)
                loss = crit(out.logits.float(), y)
            loss_sum += loss.item()
            n += y.numel()
            probs.append(torch.softmax(out.logits.float(), -1)[:, 1].cpu().numpy())
            labels.append(y.cpu().numpy())
    p = np.concatenate(probs)
    y_all = np.concatenate(labels)
    m = metrics(y_all, p)
    m["loss"] = loss_sum / n
    return m, p, y_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="chinese_roberta_L-4_H-256")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--frac", type=float, default=1.0, help="训练集采样比例，用于学习曲线")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"

    mpath = os.path.join(MODELS, args.model)
    tok = BertTokenizer.from_pretrained(mpath)
    model = BertForSequenceClassification.from_pretrained(mpath, num_labels=2).to(device)
    n_param = sum(p.numel() for p in model.parameters())

    train_rows = read_tsv("train")
    if args.frac < 1.0:
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(len(train_rows))[: int(len(train_rows) * args.frac)]
        train_rows = [train_rows[i] for i in idx]
    train_ds = Cls(train_rows, tok, args.max_len)
    val_ds = Cls(read_tsv("val"), tok, args.max_len)
    test_ds = Cls(read_tsv("test"), tok, args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": args.wd},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)
    crit = nn.CrossEntropyLoss()

    print(f"模型 {args.model} | 参数 {n_param:,} | 训练 {len(train_ds)} 条 | "
          f"{len(train_dl)}×{args.epochs} 步 | 设备 {device}", flush=True)

    history, steps_log = [], []
    best = {"macro_f1": -1}
    t0 = time.time()
    gstep = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        ep0 = time.time()
        run_loss, run_n = 0.0, 0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(input_ids=ids, attention_mask=mask, labels=y)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            gstep += 1
            run_loss += loss.item() * y.numel()
            run_n += y.numel()
            steps_log.append({"step": gstep, "loss": loss.item(),
                              "lr": sched.get_last_lr()[0], "grad_norm": float(gn)})

        val_m, _, _ = evaluate(model, val_dl, device, amp)
        rec = {"epoch": ep, "train_loss": run_loss / run_n,
               "val_loss": val_m["loss"], "val_acc": val_m["acc"],
               "val_macro_f1": val_m["macro_f1"], "val_f1_pos": val_m["f1_pos"],
               "val_f1_neg": val_m["f1_neg"], "time": round(time.time() - ep0, 1)}
        history.append(rec)
        print(f"epoch {ep}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  "
              f"acc {rec['val_acc']*100:.2f}%  macroF1 {rec['val_macro_f1']*100:.2f}%  "
              f"({rec['time']}s)", flush=True)
        if val_m["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val_m["macro_f1"], "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    # 用验证集上最好的一轮去测测试集
    model.load_state_dict(best["state"])
    test_m, test_p, test_y = evaluate(model, test_dl, device, amp)
    print(f"test  acc {test_m['acc']*100:.2f}%  macroF1 {test_m['macro_f1']*100:.2f}%  "
          f"(第 {best['epoch']} 轮)", flush=True)

    torch.save(best["state"], os.path.join(OUT, f"{args.tag}_ckpt.pt"))
    np.savez(os.path.join(OUT, f"{args.tag}_test.npz"), prob=test_p, label=test_y)

    result = {
        "tag": args.tag, "model": args.model, "params": n_param,
        "config": vars(args), "history": history, "steps": steps_log,
        "best_epoch": best["epoch"], "test": test_m,
        "total_time": round(time.time() - t0, 1), "device": device,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "n_steps": gstep,
    }
    with open(os.path.join(OUT, f"{args.tag}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved", os.path.join(OUT, f"{args.tag}_metrics.json"))


if __name__ == "__main__":
    main()
