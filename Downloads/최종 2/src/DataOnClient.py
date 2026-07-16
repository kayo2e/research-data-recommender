import re, requests
import html
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def clean_description(text: str) -> str:
    """
    설명(desc) 필드 정제 함수
    - HTML 링크, 특수문자, 개행기호 제거
    - 따옴표, 괄호, °, 불필요한 escape 문자 제거
    - Warning 블록 자동 제거
    - 공백 및 HTML 엔티티(&nbsp; 등) 정리
    """
    if not text:
        return ""

    # 0. HTML 엔티티(&nbsp;, &quot; 등) 디코딩
    text = html.unescape(text)

    # 1. Markdown / HTML 링크 제거 ([...](...))
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # 2. Warning 블록 제거 (**Warning:** ... interpolation.)
    text = re.sub(
        r"\*\*Warning:\*\*.*?(?:bilinear interpolation\.|interpolation\.)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # 3. 개행문자 및 제어문자 제거
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")

    # 4. 따옴표 및 특수 따옴표 제거
    text = text.replace("“", "").replace("”", "").replace('"', "'")

    # 5. 기타 특수문자 제거 / 정리
    text = re.sub(r"[°•●▪▶◆■□◇☆★※→←↑↓▲▼◀▶·•▶]", " ", text)

    # 6. HTML 태그(<...>) 제거
    text = re.sub(r"<.*?>", " ", text)

    # 7. 중복된 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    # 8. 의미 없는 단독 텍스트 제거
    if re.match(r"(?i)^\s*(none|null|warning|n/a)\s*$", text):
        return ""

    return text

class DataOnApiClient:
    def __init__(self, search_key, detail_key):
        self.search_key, self.detail_key = search_key, detail_key
        self.search_url = "https://dataon.kisti.re.kr/rest/api/search/dataset"
        self.model = MODEL

    def search(self, query_list, size=10):
        if isinstance(query_list, str):
            query_items = [query_list]
        else:
            query_items = query_list
        all_results = {}
        for phrase in query_items:
            clean_phrase = re.sub(r"[^\w가-힣\s]", " ", phrase)
            clean_phrase = re.sub(r"\s+", " ", clean_phrase).strip()
            if not clean_phrase:
                continue
            params = {"key": self.search_key, "query": clean_phrase, "size": size}
            try:
                r = requests.get(self.search_url, params=params, timeout=10)
                r.raise_for_status()
                for rec in r.json().get("records", []):
                    dataset_id = rec.get("svc_id", "")
                    if dataset_id not in all_results:
                        all_results[dataset_id] = {
                            "구분": "dataset",
                            "dataset_id": dataset_id,
                            "title": rec.get("dataset_title_kor") or rec.get("dataset_title_etc_main", ""),
                            # ✅ 설명 정제 적용
                            "desc": clean_description(
                                rec.get("dataset_expl_kor") or rec.get("dataset_expl_etc_main", "")
                            ),
                            "url": rec.get("dataset_lndgpg") or f"https://dataon.kisti.re.kr/search/portal/detail.do?svcId={dataset_id}"
                        }
            except Exception as e:
                print(f"⚠ '{phrase}' 검색 중 오류 발생: {e}")
                continue
        return list(all_results.values())

    def detail(self, svcid):
        r = requests.get(self.search_url, params={'key': self.detail_key, 'query': svcid, 'size': 1}, timeout=10)
        recs = r.json().get("records", [])
        if not recs:
            return None
        rec = recs[0]
        return {
            "dataset_id": rec.get("svc_id", ""),
            "title": rec.get("dataset_title_kor") or rec.get("dataset_title_etc_main", ""),
            # ✅ detail()에도 정제 함수 적용
            "desc": clean_description(
                rec.get("dataset_expl_kor") or rec.get("dataset_expl_etc_main", "")
            ),
            "url": rec.get("dataset_lndgpg") or f"https://dataon.kisti.re.kr/search/portal/detail.do?svcId={svcid}"
        }

    def batched_encode(self, texts, batch_size=8):
        """대용량 문장 리스트를 CPU에서 안전하게 배치 단위로 임베딩"""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                embs = self.model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
                all_embs.extend(embs)
            except Exception as e:
                print(f"[WARN] ⚠ 배치 인코딩 실패 ({i}~{i+len(batch)}): {e}")
        return np.array(all_embs)
