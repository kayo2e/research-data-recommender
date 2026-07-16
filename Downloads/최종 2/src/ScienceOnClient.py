import os, re, json, requests, xml.etree.ElementTree as ET
from urllib import parse
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
import torch

torch.cuda.is_available = lambda: False  
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def refresh_kisti_token(client_id, refresh_token):
    url = "https://apigateway.kisti.re.kr/tokenrequest.do"
    payload = {"client_id": client_id, "grant_type": "refresh_token", "refresh_token": refresh_token}
    try:
        #print("[INFO] 🔄 KISTI Access Token 요청 중...")
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
        token = r.json().get("access_token", "")
        #print(f"[SUCCESS] ✅ Access Token 발급 ({token[:16]}...)")
        return token
    except Exception as e:
        print("[ERROR] 🔥 토큰 발급 실패:", e)
        return None

class KistiApiClient:
    def __init__(self, client_id, refresh_token):
        self.client_id, self.refresh_token = client_id, refresh_token
        self.access_token = refresh_kisti_token(client_id, refresh_token)
        self.search_url = "https://apigateway.kisti.re.kr/openapicall.do"

    def _single_search(self, field, phrase, target="ARTI", size=10):
        """ScienceON 단일 필드 검색 (Title, KW, Abstract)"""
        query_json = parse.quote(json.dumps({field: phrase}))
        url = (
            f"{self.search_url}?client_id={self.client_id}&token={self.access_token}"
            f"&version=1.0&action=search&target={target}"
            f"&searchQuery={query_json}&rowCount={size}"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 401:
                print("[WARN] ⚠ Token 만료 → 재발급 중...")
                self.access_token = refresh_kisti_token(self.client_id, self.refresh_token)
                return self._single_search(field, phrase, target, size)
            results = []
            root = ET.fromstring(r.text)
            for record in root.findall(".//recordList/record"):
                items = {i.get("metaCode"): (i.text or "") for i in record.findall("item")}
                cn = items.get("CN", "").strip()
                title = items.get("Title", "").strip()
                desc = items.get("Abstract", "").strip()
                if not title or not cn:
                    continue
                results.append({
                    "구분": "paper",
                    "paper_id": cn,
                    "title": title,
                    "desc": desc,
                    "url": f"https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn={cn}"
                })
            return results
        except Exception as e:
            print(f"⚠ '{phrase}' ({field}) 검색 중 오류 발생: {e}")
            return []

    def search(self, keywords, target="ARTI", size=10):
        """ScienceON — KW, Title, Abstract 각각 검색 후 중복 제거"""
        if isinstance(keywords, str):
            query_items = [keywords]
        else:
            query_items = keywords
        all_results = {}
        fields = ["KW", "TI","AB"]
        for phrase in query_items:
            clean_phrase = re.sub(r"[^\w가-힣\s]", " ", phrase)
            clean_phrase = re.sub(r"\s+", " ", clean_phrase).strip()
            if not clean_phrase:
                continue
            #print(f"\n ScienceON '{clean_phrase}' 검색 시작")
            # 각 필드별로 병렬 검색 수행
            with ThreadPoolExecutor(max_workers=3) as ex:
                futures = [ex.submit(self._single_search, f, clean_phrase, target, size) for f in fields]
                results_per_field = [f.result() for f in futures]
            # 결과 병합 및 중복 제거
            for field, result_list in zip(fields, results_per_field):
                #print(f" └ {field} 검색 결과: {len(result_list)}개")
                for rec in result_list:
                    cn = rec["paper_id"]
                    if cn not in all_results:
                        all_results[cn] = rec
        results = list(all_results.values())
        #print(f"✅ ScienceON 총 {len(results)}개 논문 수집 완료 (중복 제거됨)")
        return results