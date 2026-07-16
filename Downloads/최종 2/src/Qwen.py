import re, json,requests

class QwenApi:
    def __init__(self, endpoint, token):
        self.endpoint = endpoint
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def truncate_text(self, text, max_chars=500):
        if not text:
            return ""
        return text[:max_chars] + ("..." if len(text) > max_chars else "")

    def extract_features(self, title, desc):
        truncated_title = self.truncate_text(title, 200)
        truncated_desc = self.truncate_text(desc, 800)
        user_message = (
            f"제목: {truncated_title}\n"
            f"설명: {truncated_desc}\n\n"
            "위 텍스트에서 핵심 목표(goal)와 주요 방법론(methodology)를 분석하여 아래 JSON으로 답변:\n"
            '{"goal": "텍스트", "methodology": "텍스트"}'
        )
        payload = {
            "model": "qwen3-14b",
            "messages": [{"role": "user", "content": user_message}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        try:
            r = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            # ✅ 문자열일 경우 JSON 디코딩
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    print("[WARN] ⚠ JSON 파싱 실패, 기본 구조 반환")
                    return {"goal": "", "methodology": ""}
            elif isinstance(raw, dict):
                return raw
            else:
                return {"goal": "", "methodology": ""}
        except Exception as e:
            print(f"LLM 특징 추출 실패: {e}")
            return {"goal": "", "methodology": ""}

    def generate_search_keywords(self, title: str, goal: str, methodology: str) -> list:
        """주어진 컨텍스트를 바탕으로 최적의 검색어를 생성하고 '리스트'로 반환합니다."""
        truncated_title = self.truncate_text(title, 200)
        truncated_goal = self.truncate_text(goal, 400)
        truncated_methodology = self.truncate_text(methodology, 400)
        user_message = (
            "다음은 특정 자료의 핵심 내용입니다.\n"
            f" - 제목: {truncated_title}\n"
            f" - 핵심 목표: {truncated_goal}\n"
            f" - 주요 방법론: {truncated_methodology}\n\n"
            "이 자료의 구체적인 내용에만 국한되지 말고, 이 자료가 속한 **더 넓은 상위 연구 분야(Broader Research Area)나 해결하려는 핵심 문제(Core Problem)가 무엇인지 추론**해 주세요.\n"
            "그런 다음, 그 상위 분야나 문제와 관련된 다른 중요한 논문들을 찾을 수 있는 **일반적이고 핵심적인 검색어 5개**를 생성해 주세요.\n\n"
            "예를 들어, 입력 자료가 'Social-GAN' 모델에 대한 것이라면, 'Trajectory Prediction', 'Generative Adversarial Network', 'Human-Object Interaction' 같은 더 일반적인 키워드를 생성해야 합니다.\n"
            "결과는 쉼표(,)로 구분된 키워드 목록으로만 답변해 주세요."
        )
        payload = {
            "model": "qwen3-14b",
            "messages": [
                {"role": "system", "content": "You are a research assistant skilled at identifying the core research field of a paper and generating abstract keywords."},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.0
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            keywords_str = response.json()["choices"][0]["message"]["content"]
            clean_keywords = re.sub(r"<think>.*?</think>\s*", "", keywords_str, flags=re.DOTALL).strip()
            keywords_list = [kw.strip() for kw in clean_keywords.split(',') if kw.strip()]
            return keywords_list
        except Exception as e:
            print(f"LLM 검색어 생성 실패: {e}. 입력 데이터의 제목을 대체 검색어로 사용합니다.")
            return [title]

    def group_keywords_into_phrases(self, keyword_list: list) -> list:
        """단어 리스트를 받아 의미있는 구(phrase) 리스트로 재조합합니다."""
        if not keyword_list or len(keyword_list) <= 5:
            return keyword_list
        user_message = (
            "다음은 아이디어 브레인스토밍을 통해 생성된 개별 키워드 목록입니다:\n"
            f" - [ {', '.join(keyword_list)} ]\n\n"
            "이 목록의 단어들을 조합하여, 의미 있는 5개 내외의 전문 기술 용어 또는 연구 분야(multi-word technical phrases or research areas)로 재구성해 주세요.\n"
            "각각의 구(phrase)는 쉼표로 구분하여 목록 형태로만 답변해 주세요.\n"
            "불필요하거나 일반적인 단어(for, and, a 등)는 제외해도 좋습니다.\n"
            "예를 들어, ['Pre-trained', 'Language', 'Models', 'Clinical', 'Text']가 입력되면, 'Pre-trained Language Models', 'Clinical Text Analysis' 와 같이 재구성해야 합니다."
        )
        payload = {
            "model": "qwen3-14b",
            "messages": [
                {"role": "system", "content": "You are an expert research assistant who can group individual keywords into meaningful technical phrases."},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.0
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            grouped_keywords_str = response.json()["choices"][0]["message"]["content"]
            clean_keywords = re.sub(r"<think>.*?</think>\s*", "", grouped_keywords_str, flags=re.DOTALL).strip()
            final_list = [kw.strip() for kw in clean_keywords.split(',') if kw.strip()]
            return final_list
        except Exception as e:
            print(f"LLM 키워드 그룹핑 실패: {e}. 그룹핑 없이 기존 키워드를 사용합니다.")
            return keyword_list

    def generate_recommendation_reason(self, input_data, recommended_data):
        input_title = self.truncate_text(input_data.get("title", ""), 200)
        rec_title = self.truncate_text(recommended_data.get("title", ""), 200)
        user_message = (
            f"입력 데이터 제목: '{input_title}'\n"
            f"추천 데이터 제목: '{rec_title}'\n\n"
            "두 데이터의 관련성을 **2~3 문장의 상세한 설명**으로 생성해 주세요.\n"
            "첫 문장에서는 **핵심적인 공통점이나 주제**를 명확히 언급해 주세요.\n"
            "이어지는 문장에서는 추천 데이터가 입력 데이터의 관점에서 **어떤 점에서 더 구체적인지, 혹은 어떤 새로운 정보를 제공하는지** 구체적으로 서술하여 추천의 가치를 설명해 주세요."
        )
        payload = {
            "model": "qwen3-14b",
            "messages": [
                {"role": "system", "content": "You are a helpful research assistant who explains the connections between academic papers."},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.0
        }
        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            reason = response.json()["choices"][0]["message"]["content"]
            return re.sub(r"<think>.*?</think>\s*", "", reason, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"추천 사유 생성 실패: {e}")
            return "추천 사유를 생성할 수 없습니다."