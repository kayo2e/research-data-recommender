# README.md

# **sLLM 기반 논문·데이터셋 추천 에이전트**

**DataON & ScienceON의 방대한 연구 데이터 속에서 의미적으로 가장 연관성 높은 논문과 데이터셋을 추천하는 AI 에이전트입니다.**

## **1. 프로젝트 소개 (Introduction)**

본 프로젝트는 키워드 검색의 한계를 넘어, 소규모 언어모델(sLLM)을 활용해 연구 데이터의 의미적 맥락을 깊이 있게 이해하고 관련 자료를 추천합니다. 사용자가 입력한 데이터의 '핵심 목표'와 '주요 방법론'을 추론하여, 이를 바탕으로 개인화된 추천 결과를 제공함으로써 연구 데이터의 활용 가치를 극대화하는 것을 목표로 합니다.

## 2. 실행 환경 (Execution Environment)

본 모델은 아래의 환경에서 개발 및 테스트되었습니다.

- **OS:** Windows 11
- **Hardware:**
    - CPU: Intel64 Family 6 Model 191 Stepping 2, GenuineIntel
    - RAM: 16GB 이상 (32GB 권장)
    - GPU: None
- **Software:**
    - NVIDIA Driver: `572.16`
    - CUDA Version: `12.8`
    - Python: `3.12.8`
- **Software:**
    - sentence-transformers: `5.1.0`
    - langchain-core: `0.3.79`
    - numpy: `2.3.3`
    - tqdm: `4.67.1`

## **3. 주요 기능 (Features)**

- **의미 기반 추천**: sLLM을 이용해 데이터의 핵심 주제와 맥락을 분석하여 콘텐츠를 추천합니다.
- **다차원적 정보 제공**: 관련성 점수, 상세 추천 사유, '강추/추천/참고' 등급을 함께 제공하여 사용자의 이해를 돕습니다.
- **경량화 및 고효율**: 중저사양의 하드웨어(e.g., RTX 3060)에서도 빠른 응답 속도를 보장하여 실제 서비스 적용 가능성을 높였습니다.

## **4. 기술 스택 (Tech Stack)**

- **Language**: `Python 3.12.8`
- **AI/ML Models**:
    - `qwen3-14b` (sLLM for Reranking & Reasoning)
    - `paraphrase-multilingual-MiniLM-L12-v2` (Embedding Model for Initial Filtering)
- **APIs**:
    - `DataON API`
    - `ScienceON API`
    - `Gemini API`  (평가용)
- **Key Libraries**:
    - `Transformers`, `Sentence-Transformers`, `PyTorch`
    - `langchain-google-genai, langchain-core`
    - `Requests`, `numpy`, `pandas`, `python-dotenv`, `tqdm`

## **5. 실행 방법 (Usage)**

**필수 라이브러리 설치:**

프로젝트 실행에 필요한 라이브러리들이 설치되어 있어야 합니다. 

`requirements.txt` 파일을 이용하여 pip으로 설치해주세요.

```python
pip install -r requirements.txt
```

### **5.2. 프로그램 실행**

모든 준비가 완료되면, main.ipynb 파일을 실행합니다.

```python
# 1. 메인 파일 실행
main.ipynb 파일 실행

# 2. Qwen 기반 추천 시스템 실행 후 dataset_id 입력
dataset_id를 입력하세요: [테스트할 DataON의 dataset_id]
```

## **6. 프로젝트 구조 (Project Structure)**

```python
.
├── requirements.txt
├── main.ipynb               # 개발 및 테스트 과정을 담은 ipynb 폴더
├── data/
│   ├── dataon_satellite.json      # 평가용 데이터셋
│   ├── qwen_satellite_psudo_gt/
│   ├── ├── qwen_judge_checkpoint.json # Qwen 모델 결과 저장 체크포인트
│   └── └── qwen_judge_results.json    # Qwen 모델 결과 저장 최종
├── outputs/
│   ├─ recommend_result.csv  # 추천 클라이언트 결과 파일
├── src/                     # 핵심 소스 코드 폴더
│   ├── AES256Util.py        # 사이언스온 토큰 발급 Util
│   ├── TokenSample.py       # 사이언스온 토큰 발급 메인 함수
│   ├── DataOnClient.py      # 데이터온 클라이언트
│   ├── ScienceOnClient.py   # 사이언스온 클라이언트
│   ├── Qwen.py              # Qwen 모델 실행
│   ├── qwen3_14b_token      # Qwen 모델 토큰
│   ├── dataon_data.py       # 평가 데이터셋 생성
│   ├── Gemini.py            # Gemini 클라이언트 (0~3점 채점용)
│   ├── RecommendAgent.py    # 메인 추천 클라이언트
│   ├── save_result.py       # 답변 저장용
│   └── eval_metrics.py      # 정량 지표 계산
└── README.md
```

## **7. 동작 원리 (How it Works)**

본 에이전트는 **LLM과 임베딩 모델을 결합한 2단계 하이브리드 파이프라인**으로 동작합니다.

1. **1단계: 후보군 생성 (Candidate Generation)**
    - **특징 추출**: `qwen3-14b`가 입력 데이터의 제목/설명에서 '핵심 목표'와 '주요 방법론'을 추출합니다.
    - **검색어 생성 (2-Step Chain)**:
2. **키워드 브레인스토밍**: 추출된 특징을 기반으로 더 넓은 상위 연구 분야와 관련된 핵심 키워드를 생성합니다.
3. **키워드 그룹핑**: 생성된 개별 키워드들을 조합하여 의미 있는 전문 용어 또는 연구 분야로 재구성하여 검색 효율을 높입니다.
    - **1차 필터링**: 임베딩 모델로 API 검색 후보군과 입력 데이터 간의 코사인 유사도를 계산하여 상위 10~15개의 후보를 빠르게 선정합니다.
4. **2단계: 최종 추천 (Final Recommendation)**
    - **재정렬 (Re-ranking)**: `qwen3-14b`가 1차 필터링된 후보들의 문맥적 관련성(목표, 방법론 중심)을 1~10점으로 다시 평가합니다.
    - **최종 점수 산출**: `유사도 점수(60%)`와 `재정렬 점수(40%)`를 가중 합산하여 최종 순위를 결정합니다.
    - **추천 사유 생성**: `qwen3-14b`가 각 추천 결과에 대한 구체적인 연관성을 설명하는 문장을 생성합니다.

## **8. 성능 평가 (Evaluation)**

- **평가 지표**: 추천 결과의 순위와 정확도를 종합적으로 측정하기 위해 아래 지표를 사용합니다.
    - `nDCG@10`: 추천 목록 상위 10개의 순서와 관련성을 함께 고려하는 지표
    - `MRR@10`: 상위 10개 중 첫 번째 정답의 순위를 평가하는 지표
    - `Recall@5`: 상위 5개 내 실제 관련 항목 포함 비율
    - `Precision@5`: 상위 5개 중 정확히 맞힌 추천의 비율
    - `F1@5`: Precision과 Recall의 균형을 나타내는 평균 지표

## **9. 향후 계획 (Future Work)**

- **대화형 챗봇 인터페이스 도입**: 추천 결과에 대한 후속 질문 및 동적 조건 조정 기능 추가
- **다국어 지원**: 다국어 모델을 적용하여 글로벌 논문 및 데이터셋으로 추천 범위 확장
- **사용자 맞춤형 추천**: 사용자 검색 이력 및 관심 분야를 학습하여 개인화된 추천 기능 구현