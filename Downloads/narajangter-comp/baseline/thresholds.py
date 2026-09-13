# 24개 항목 중 절반 가까이가 "고시금액"/"수의계약 기준금액" 같은 외부 임계값과 배정예산금액·
# 입찰추정가격을 비교해야 판정 가능한데, 베이스라인 프롬프트엔 그 숫자 자체가 없다. 모델이
# 기억해서 맞히길 기대하는 대신, 여기서 결정론적으로 계산해 프롬프트에 "이미 계산된 사실"로
# 박아준다(neuro-symbolic 패턴) — 법령 원문에서 실제 조문을 찾아 확인한 값만 담는다.
from __future__ import annotations

import csv
import io
import os
import re
from typing import Optional

# meta.세부품명번호목록은 "품명[코드], 품명[코드], ..." 형식의 단일 문자열로 온다(리스트가 아님) —
# 코드만 정규식으로 뽑아낸다.
_PRODUCT_CODE_RE = re.compile(r"\[(\d+)\]")

# ---- A. 고시금액(WTO 정부조달협정 기준) ----
# 국가계약법 시행령 제2조제3호 "고시금액=재정경제부장관이 고시한 금액"은 어느 조약을 쓸지
# 특정하지 않지만, 시행령 제21조 등 일반 조문에서 그냥 "고시금액"이라 쓸 때는 실무상 WTO
# 정부조달협정(GPA) 기준값을 가리키는 게 표준 해석이다(법령패키지의 "국가를 당사자로 하는
# 계약에 관한 법률 등의 재정경제부장관이 정하는 고시금액.txt" 1-가 항).
GOSI_AMOUNT_GOODS_SERVICES = 230_000_000  # 물품 및 용역
GOSI_AMOUNT_CONSTRUCTION = 8_800_000_000  # 공사

# ---- C-1. 지방계약 "2인 이상 견적서 제출 수의계약" 기준금액 ----
# (지방자치단체 입찰 및 계약 집행기준 제5장 제3절 1. 금액기준에 따른 2인 이상 견적서 제출
# 수의계약 — 원문 표 그대로) v6~v8의 "고시금액 미만 지역제한"이 실제로 기대는 임계값은 이
# 표다(고시금액 자체가 아니라, 소액수의계약 범위 안에서 지역을 제한할 수 있는 한도).
TWO_QUOTE_LIMIT = {
    "종합공사": 400_000_000,
    "전문공사": 200_000_000,
    "전기등그밖의공사": 160_000_000,
    "용역물품기타": 100_000_000,
}

# ---- C-2. 지방계약 "1인 견적서 제출 가능" 기준금액 ----
ONE_QUOTE_LIMIT_DEFAULT = 20_000_000
ONE_QUOTE_LIMIT_PREFERRED = 50_000_000  # 여성/장애인/사회적기업 등 우대기업

# ---- C-3. 공동수급 최소지분율 ((계약예규) 공동계약운용요령 제9조 제5항) ----
def min_joint_venture_share(방식: Optional[str], 추정가격: Optional[int]) -> Optional[float]:
    """공동수급 방식별 법정 최소지분율(%). 방식을 모르면 None(판단 보류)."""
    if not 방식:
        return None
    if "공동이행" in 방식:
        # 원칙 10% 이상, 단 추정가격 1,000억원 이상 공사는 5% 이상
        if 추정가격 is not None and 추정가격 >= 100_000_000_000:
            return 5.0
        return 10.0
    if "주계약자" in 방식:
        return 5.0
    if "분담이행" in 방식:
        return None  # 분담이행방식은 최소지분율 규정 자체가 없음(인원수 제한만 있음)
    return None


# ---- D. 중소기업자간 경쟁제품 세부품명 조회 ----
_COMPETITIVE_PRODUCT_CODES: set[str] | None = None


def _load_competitive_product_codes(data_dir: str) -> set[str]:
    path = os.path.join(data_dir, "법령패키지", "중기부고시", "중기부고시_경쟁제품_세부품명.csv")
    codes: set[str] = set()
    if not os.path.exists(path):
        return codes
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = (row.get("세부품명번호") or "").strip()
            if code:
                codes.add(code)
    return codes


def is_competitive_product(세부품명번호: str, data_dir: str) -> bool:
    global _COMPETITIVE_PRODUCT_CODES
    if _COMPETITIVE_PRODUCT_CODES is None:
        _COMPETITIVE_PRODUCT_CODES = _load_competitive_product_codes(data_dir)
    return 세부품명번호 in _COMPETITIVE_PRODUCT_CODES


def format_amount(amount: int) -> str:
    return f"{amount:,}원"


def build_threshold_notes(meta: dict, data_dir: str) -> str:
    """meta 하나로부터 "이미 계산된 사실" 문장 목록을 만든다 — 모델이 임계값을 기억할
    필요 없이 이 문장만 보고 비교 결과를 확인하면 되도록."""
    notes: list[str] = []

    # meta.조항호내용은 공고가 스스로 밝힌 근거 조항/구간이다(예: "추정가격 1억원 미만
    # 물품·용역(소기업, 소상공인, 벤처기업, 창업자)") — v13~v18(중소기업 제한 계열) 판정에
    # 임계값 계산보다도 더 직접적인 신호라서, 21개 메타 필드 중 하나로 묻히지 않게 따로
    # 강조한다. 다만 이 필드 자체가 오기재일 수도 있으니(예: v24처럼 meta-본문 불일치가
    # 점검 대상인 항목도 있음) "참고"로만 제시하고 본문 내용과 대조하라고 명시한다.
    clause = meta.get("조항호내용")
    if clause and clause not in ("미기재", "미입력"):
        notes.append(
            f"[참고] 이 공고가 스스로 밝힌 근거 조항/구간: '{clause}'. "
            f"v13~v18 판정 시 이게 실제 배정예산금액/입찰추정가격 구간, 본문의 참가자격 제한 "
            f"내용과 실제로 맞는지 대조하세요(단, meta 오기재 가능성도 있으니 본문이 우선)."
        )

    def to_amount(v) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(str(v).replace(",", "").strip())
        except ValueError:
            return None

    budget = to_amount(meta.get("배정예산금액"))
    est = to_amount(meta.get("입찰추정가격"))
    reference_amount = est if est is not None else budget

    if reference_amount is not None:
        cmp_word = "이상" if reference_amount >= GOSI_AMOUNT_GOODS_SERVICES else "미만"
        notes.append(
            f"[참고] 물품·용역 기준 고시금액은 {format_amount(GOSI_AMOUNT_GOODS_SERVICES)}입니다. "
            f"이 공고의 추정가격/배정예산({format_amount(reference_amount)})은 고시금액 {cmp_word}입니다."
        )
        two_quote_used = TWO_QUOTE_LIMIT["용역물품기타"]
        cmp_word2 = "이하" if reference_amount <= two_quote_used else "초과"
        notes.append(
            f"[참고] 용역·물품 2인 이상 견적 수의계약 기준금액은 {format_amount(two_quote_used)}입니다. "
            f"이 공고 금액은 그 기준 {cmp_word2}입니다(v6~v8의 소액 지역제한 판단 시 참고)."
        )
        cmp_word3 = "이하" if reference_amount <= ONE_QUOTE_LIMIT_DEFAULT else "초과"
        notes.append(
            f"[참고] 1인 견적 제출 가능 기준금액은 {format_amount(ONE_QUOTE_LIMIT_DEFAULT)}입니다"
            f"(우대기업은 {format_amount(ONE_QUOTE_LIMIT_PREFERRED)}). 이 공고 금액은 일반 기준 {cmp_word3}입니다."
        )

    joint_method = meta.get("공동도급구성방식")
    if joint_method:
        share = min_joint_venture_share(joint_method, reference_amount)
        if share is not None:
            notes.append(
                f"[참고] 공동수급 방식({joint_method})에서 법정 최소지분율은 {share}% 이상입니다. "
                f"공고문에 이보다 낮은 지분율을 요구하면 위반입니다(v21)."
            )

    product_field = meta.get("세부품명번호목록")
    if product_field and product_field not in ("미기재", "미입력"):
        # "품명[코드], 품명[코드], ..." 형식 — 코드만 뽑아 CSV와 대조한다.
        codes = _PRODUCT_CODE_RE.findall(str(product_field))
        for code in codes:
            flag = "예" if is_competitive_product(code, data_dir) else "아니오"
            notes.append(f"[참고] 세부품명번호 {code}는 중소기업자간 경쟁제품입니다: {flag} (v9~v13 참고).")

    award_method = meta.get("낙찰방법") or ""
    is_negotiated = "협상" in award_method
    notes.append(
        f"[참고] 이 공고의 낙찰방법은 '{award_method or '미기재'}'입니다 — "
        f"v22는 협상에 의한 계약일 때만, v23은 협상에 의한 계약 + 지방계약법일 때만 적용됩니다"
        f"(현재 협상 여부: {'예' if is_negotiated else '아니오'})."
    )

    return "\n".join(notes)
