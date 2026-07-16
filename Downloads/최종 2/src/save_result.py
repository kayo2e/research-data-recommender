import csv
import os
from datetime import datetime

def save_recommendations_to_csv(result, save_dir="./outputs"):
    """추천 결과(result 리스트)를 CSV 파일로 저장"""
    os.makedirs(save_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"recommend_result.csv")

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["구분", "제목", "설명", "점수", "추천 사유", "URL"])
        for item in result:
            writer.writerow([
                item.get("구분", ""),
                item.get("title", ""),
                item.get("desc", ""),
                item.get("score", ""),
                item.get("추천 사유", ""),
                item.get("url", "")
            ])

    print(f"✅ 추천 결과 CSV 저장 완료: {save_path}")
    return save_path
