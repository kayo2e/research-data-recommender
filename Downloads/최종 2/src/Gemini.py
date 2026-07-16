# -*- coding: utf-8 -*-
"""
[LLM 기반 pseudo-Judge 단계 – Gemini Flash Version]
- ✅ main.py에서 설정값 주입받는 구조로 리팩터링
- ✅ 하드코딩 제거, 재사용성 향상
"""

import os
import re
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


# =========================================================
# Gemini 안전
# =========================================================
def safe_gemini_call(prompt, key, sys_text="You are an expert evaluator.", model="gemini-2.5-flash", max_retry=3):
    """Gemini 모델 호출 (단일 키 기반, 호출 실패 시 예외 발생)"""
    os.environ["GEMINI_API_KEY"] = key
    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.0, google_api_key=key)
        resp = llm.invoke([SystemMessage(content=sys_text), HumanMessage(content=prompt)])
        return getattr(resp, "content", str(resp))
    except Exception as e:
        raise RuntimeError(f"[Gemini Error] {e}")


# =========================================================
# Judge 프롬프트 정의
# =========================================================
def llm_judge(query_title, cand_title, cand_reason, api_keys, model_name, delay):
    """추천 결과 1건에 대해 0~3점 점수 평가"""
    prompt = f"""
    You are an expert academic evaluator.
    Evaluate whether the following recommendation is logical and relevant.

    [Dataset Title]
    {query_title}

    [Recommended Paper Title]
    {cand_title}

    [Recommendation Reason]
    {cand_reason}

    Scoring Criteria:
    - 0: Irrelevant or illogical.
    - 1: Slightly related but vague.
    - 2: Clearly related and logical.
    - 3: Highly relevant and precise.

    Output strictly in JSON:
    {{
      "reason": "Evaluation reasoning",
      "score": number (0~3)
    }}
    """

    for i, key in enumerate(api_keys):
        try:
            raw = safe_gemini_call(prompt, key, model=model_name)
            break
        except Exception as e:
            print(f"[WARN] 키 {i+1}/{len(api_keys)} 실패 → 다음 키 시도 ({e})")
            time.sleep(3)
            raw = "{}"

    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return 0.0, raw.strip()
    try:
        parsed = json.loads(m.group(0))
        return float(parsed.get("score", 0)), parsed.get("reason", "")
    except:
        return 0.0, raw.strip()


# =========================================================
# 메인 함수
# =========================================================
def run_llm_judge(recomm_file, out_dir, api_keys, model_name="gemini-2.5-flash", llm_delay=0.2):
    """추천 결과 JSON을 불러와 LLM으로 평가 후 저장"""
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "2qwen_llm_judge_results.json")
    ckpt_file = os.path.join(out_dir, "2qwen_judge_checkpoint.json")

    with open(recomm_file, "r", encoding="utf-8") as f:
        rec_data = json.load(f)

    judged = []
    if os.path.exists(ckpt_file):
        try:
            judged = json.load(open(ckpt_file, "r", encoding="utf-8"))
            print(f"[INFO] 체크포인트 복원됨: {len(judged)}개")
        except:
            judged = []

    done_ids = {j["svc_id"] for j in judged}
    todo = [r for r in rec_data if r["svc_id"] not in done_ids]

    for rec in tqdm(todo, desc="Pseudo-Judge 평가 중", ncols=100):
        q_title = rec["title"]
        results = []
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(llm_judge, q_title, c["title"], c["추천 사유"], api_keys, model_name, llm_delay): c
                for c in rec["recommendations"]
            }
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    score, reason = fut.result()
                except Exception as e:
                    score, reason = 0.0, f"(error) {e}"
                c["judge_score"] = score
                c["judge_reason"] = reason
                results.append(c)
                time.sleep(llm_delay)

        judged.append({
            "svc_id": rec["svc_id"],
            "title": q_title,
            "evaluated": results
        })

        json.dump(judged, open(ckpt_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[CHECKPOINT] {len(judged)}개 평가 저장 중...")

    json.dump(judged, open(out_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n 최종 평가 결과 저장 완료: {out_file}")
    return out_file
