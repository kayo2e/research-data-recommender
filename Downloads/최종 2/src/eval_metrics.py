import json
import numpy as np
from tqdm import tqdm

def dcg(scores, k):
    """Discounted Cumulative Gain"""
    scores = np.asarray(scores, dtype=float)[:k]
    if scores.size:
        return np.sum((2 ** scores - 1) / np.log2(np.arange(2, scores.size + 2)))
    return 0.0

def ndcg(scores, k):
    """Normalized DCG"""
    ideal = dcg(sorted(scores, reverse=True), k)
    return dcg(scores, k) / ideal if ideal > 0 else 0.0

def mrr(scores, k):
    """Mean Reciprocal Rank"""
    for i, s in enumerate(scores[:k]):
        if s >= 2.0:  # 2점 이상이면 ‘관련 있음’으로 간주
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(scores, k, threshold=2.0):
    """Recall@k: 상위 k개 중 관련 항목 비율"""
    relevant = sum(s >= threshold for s in scores[:k])
    total_rel = sum(s >= threshold for s in scores)
    return relevant / total_rel if total_rel > 0 else 0.0

def precision_at_k(scores, k, threshold=2.0):
    """Precision@k: 상위 k개 중 관련 항목의 비율"""
    relevant = sum(s >= threshold for s in scores[:k])
    return relevant / k

def f1_at_k(scores, k, threshold=2.0):
    """F1@k: Precision과 Recall의 조화 평균"""
    prec = precision_at_k(scores, k, threshold)
    rec = recall_at_k(scores, k, threshold)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def save_metrics(judge_file, k=5, threshold=2.0):
    """
    Judge 결과 JSON 파일을 불러와 NDCG, MRR, Recall, Precision, F1 평균을 계산 후 저장

    Parameters
    ----------
    judge_file : str
        LLM Judge 결과 JSON 파일 경로
    k : int
        평가할 상위 k 값 (기본=5)
    threshold : float
        '관련 있음'으로 간주할 점수 기준 (기본=2.0)
    """
    with open(judge_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ndcg_scores, mrr_scores, recall_scores, precision_scores, f1_scores = [], [], [], [], []
    all_scores = []

    for d in tqdm(data, desc="정량평가 중", ncols=100):
        evaluated = d.get("evaluated") or d.get("recommendations") or []
        scores = [float(r.get("judge_score", 0) or 0) for r in evaluated]
        if not scores:
            continue

        # 0~1 스케일로 정규화 (3점 만점 기준)
        norm_scores = [s / 3.0 for s in scores]
        all_scores.extend(norm_scores)

        ndcg_scores.append(ndcg(norm_scores, k=k))
        mrr_scores.append(mrr(scores, k=k))
        recall_scores.append(recall_at_k(scores, k=k, threshold=threshold))
        precision_scores.append(precision_at_k(scores, k=k, threshold=threshold))
        f1_scores.append(f1_at_k(scores, k=k, threshold=threshold))

    # 평균 계산
    metrics = {
        "mean_ndcg": float(np.mean(ndcg_scores)),
        "mean_mrr": float(np.mean(mrr_scores)),
        "mean_recall": float(np.mean(recall_scores)),
        "mean_precision": float(np.mean(precision_scores)),
        "mean_f1": float(np.mean(f1_scores)),
        "mean_score": float(np.mean(all_scores)),
        "std_score": float(np.std(all_scores))
    }


    #print(f"정량평가 지표 저장 완료: {out_path}")
    return metrics