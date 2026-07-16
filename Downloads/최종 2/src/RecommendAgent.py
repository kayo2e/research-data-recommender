import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import util
from DataOnClient import DataOnApiClient
from ScienceOnClient import KistiApiClient
from Qwen import QwenApi

class RecommendationAgent:
    def __init__(self, dataon_keys, kisti_info, qwen_info):
        self.dataon = DataOnApiClient(dataon_keys['search_key'], dataon_keys['detail_key'])
        self.kisti = KistiApiClient(kisti_info['client_id'], kisti_info['refresh_token'])
        self.qwen = QwenApi(qwen_info['endpoint'], qwen_info['token'])
        self.model = self.dataon.model

    def recommend(self, dataset_id, top_k_final=5):
        data = self.dataon.detail(dataset_id)
        if not data:
            print("❌ dataset_id를 찾을 수 없습니다.")
            return []
        print(f"\n[INFO] '{data['title']}' 기반 추천 시작...")

        # === Qwen으로 특징·키워드 생성 ===
        features = self.qwen.extract_features(data["title"], data["desc"])
        kws = self.qwen.generate_search_keywords(data["title"], features.get("goal", ""), features.get("methodology", ""))
        print(" - 생성된 키워드:", kws)

        # === DataON 후보 ===
        dataon_cands = self.dataon.search(kws, size=15)
        dataon_cands = [
            d for d in dataon_cands
            if d.get("dataset_id") != dataset_id and d.get("title", "").strip().lower() != data["title"].strip().lower()
        ]

        # === ScienceON 후보 ===
        sci_cands = self.kisti.search(kws, target="ARTI", size=15)

        if not dataon_cands and not sci_cands:
            print("❌ 후보 없음.")
            return []

        # === 유사도 계산 ===
        q_emb = self.model.encode(f"{data['title']} {data['desc']}", convert_to_tensor=False)
        all_cands = dataon_cands + sci_cands

        # === 설명이 부실한 항목 필터링 ===
        def is_valid_description(text):
            if not text or len(text.strip()) < 50:
                return False
            t = text.strip().lower()
            if t in ["...", "none", "null", "해당 없음"]:
                return False
            if re.match(r"^[\.\s]*$", t):
                return False
            return True

        # 설명이 충분한 후보만 남김
        all_cands = [c for c in all_cands if is_valid_description(c.get("desc", ""))]
        if not all_cands:
            print("⚠ 설명이 충분한 후보가 없어 추천을 생성할 수 없습니다.")
            return []

        c_texts = [f"{c['title']} {c['desc']}" for c in all_cands]
        c_emb = self.dataon.batched_encode(c_texts, batch_size=8)
        sims = util.cos_sim(np.array([q_emb]), np.array(c_emb))[0]

        for c, s in zip(all_cands, sims):
            c["score"] = round(float(s) * 100, 2)

        # === DataON, ScienceON 각각 상위 N개 추출 ===
        dataon_sorted = sorted([c for c in all_cands if c["구분"] == "dataset"], key=lambda x: x["score"], reverse=True)
        sci_sorted = sorted([c for c in all_cands if c["구분"] == "paper"], key=lambda x: x["score"], reverse=True)

        # 최소 1개씩 확보
        final = []
        if dataon_sorted:
            final.append(dataon_sorted[0])
        if sci_sorted:
            final.append(sci_sorted[0])

        # 나머지는 점수순으로 채우기 (중복제거)
        combined_sorted = sorted(all_cands, key=lambda x: x["score"], reverse=True)
        for c in combined_sorted:
            if c not in final and len(final) < top_k_final:
                final.append(c)

        # === LLM 추천 사유 병렬 생성 ===
        with ThreadPoolExecutor(max_workers=2) as ex:
            final = list(ex.map(lambda c: {**c, "추천 사유": self.qwen.generate_recommendation_reason(data, c)}, final))

        # === 점수 기반 추천 수준(Level) 부여 ===
        for c in final:
            s = c.get("score", 0)
            if s >= 80:
                c["level"] = "강추"
            elif s >= 60:
                c["level"] = "추천"
            elif s >= 40:
                c["level"] = "참고"
            else:
                c["level"] = "낮은 관련도"

        final = sorted(final, key=lambda x: x["score"], reverse=True)
        return final