#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dev_labels.csv 기준 자가채점 — 리더보드와 같은 지표(24개 항목 각각의 위반 클래스 F1의
단순 평균 = macro-F1)를 로컬에서 미리 확인한다. 실제 채점 데이터의 구성비는 dev와 다르다고
공지돼 있어(README §2) 여기 점수가 리더보드 점수를 그대로 예측하진 않지만, "이번 변경이
방향이 맞는지"는 이걸로 먼저 걸러낼 수 있다 — 하루 1회뿐인 실제 제출을 아끼는 용도.

사용법
  python3 score.py output/submission.csv                  # dev_labels.csv와 비교
  python3 score.py output/submission.csv --labels other.csv
  python3 score.py output/submission.csv --by-item         # 항목별 표 출력
"""
from __future__ import annotations

import argparse
import csv
import io
from typing import Dict, List, Tuple

ITEMS = [f"v{i}" for i in range(1, 25)]


def load_csv(path: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["id"]] = row
    return rows


def item_f1(pred: Dict[str, Dict[str, str]], gold: Dict[str, Dict[str, str]], item: str) -> Tuple[float, int, int, int]:
    tp = fp = fn = 0
    for id_, gold_row in gold.items():
        g = gold_row.get(item, "0").strip()
        p = (pred.get(id_) or {}).get(item, "0").strip()
        g_hit, p_hit = g == "1", p == "1"
        if p_hit and g_hit:
            tp += 1
        elif p_hit and not g_hit:
            fp += 1
        elif not p_hit and g_hit:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, tp, fp, fn


def score(submission_path: str, labels_path: str, by_item: bool) -> float:
    pred = load_csv(submission_path)
    gold = load_csv(labels_path)

    missing = set(gold) - set(pred)
    if missing:
        print(f"[경고] submission에 없는 id {len(missing)}건 — 위반여부 전부 0으로 간주: "
              f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")

    f1s: List[float] = []
    rows = []
    for item in ITEMS:
        f1, tp, fp, fn = item_f1(pred, gold, item)
        f1s.append(f1)
        rows.append((item, f1, tp, fp, fn))

    if by_item:
        print(f"{'항목':>4}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
        for item, f1, tp, fp, fn in rows:
            print(f"{item:>4}  {f1:>6.3f}  {tp:>4}  {fp:>4}  {fn:>4}")
        print()

    macro_f1 = sum(f1s) / len(f1s)
    print(f"macro-F1 ({len(gold)}건 기준): {macro_f1:.4f}")
    return macro_f1


def main() -> int:
    ap = argparse.ArgumentParser(description="dev_labels.csv 기준 자가채점(macro-F1)")
    ap.add_argument("submission", help="채점할 submission.csv 경로")
    ap.add_argument("--labels", default="dev_labels.csv", help="정답 라벨 CSV (기본: dev_labels.csv)")
    ap.add_argument("--by-item", action="store_true", help="24개 항목별 F1 표도 함께 출력")
    a = ap.parse_args()
    score(a.submission, a.labels, a.by_item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
