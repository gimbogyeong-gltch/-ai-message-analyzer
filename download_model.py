# download_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "snunlp/KR-FinBERT"  # 사용할 허깅페이스 모델 이름
OUTPUT_DIR = "model_output"       # 저장할 폴더 이름

# 1) 토크나이저 내려받아 저장
print("토크나이저 다운로드 중…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

# 2) 모델 가중치 내려받아 저장
print("모델 가중치 다운로드 중…")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.save_pretrained(OUTPUT_DIR)

print(f"✅ '{OUTPUT_DIR}' 폴더가 생성되고 모델 파일이 저장되었습니다.")
