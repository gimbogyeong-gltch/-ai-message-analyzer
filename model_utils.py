# model_utils.py

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 자원 사용량 최적화를 위한 설정
torch.set_num_threads(2)  # Intel Core i5-6200U는 2코어/4스레드

# 메모리 사용량 확인 및 관리 함수
def check_memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

# 🧩 커스텀 Dataset 클래스
class MessageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        premise = f"'{row['relationship']}' 관계에서 메시지가 적합한가요?"
        hypothesis = row['message']

        enc = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }

# 데이터 로드 & 전처리
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8")
    df = df.rename(columns={
        "문장": "message",
        "관계": "relationship",
        "적절도": "label"
    })

    # 1) 문자열 공백 제거
    df["label"] = df["label"].astype(str).str.strip()
    # 2) 숫자로 변환 불가능한 값은 NaN 처리
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    # 3) NaN(헤더나 깨진 값) 행들 통째로 제거
    df = df.dropna(subset=["label"])
    # 4) 안전하게 정수형으로 캐스팅
    df["label"] = df["label"].astype(int)

    required = ["message", "relationship", "label"]
    if not all(col in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        raise ValueError(f"CSV에 누락된 컬럼: {missing}")

    print(f"✅ 데이터 로드 완료: {len(df)}개 샘플")
    return df

# 정확도 계산 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 간단 오버샘플링
def simple_oversample(X, y):
    tmp = pd.concat([X, pd.Series(y, name='label')], axis=1)
    counts = tmp['label'].value_counts()
    max_n = counts.max()
    frames = []
    for val, cnt in counts.items():
        dfc = tmp[tmp['label'] == val]
        mul = max_n // cnt
        rem = max_n % cnt
        oversampled = pd.concat([dfc]*mul + [dfc.sample(rem, replace=True)])
        frames.append(oversampled)
    all_df = pd.concat(frames).sample(frac=1).reset_index(drop=True)
    return all_df[X.columns], all_df['label']

# 🏋️ 모델 학습 함수
def train_model(
    train_file,
    model_name: str = 'snunlp/KR-FinBERT',
    batch_size: int = 2,
    epochs: int = 2,
    output_dir: str = "./model_output"
):
    df = load_data(train_file)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train = simple_oversample(
        train_df[['message','relationship']],
        train_df['label']
    )
    train_df = pd.concat([X_train, pd.Series(y_train, name='label')], axis=1)

    print(f"📊 전처리 후 메모리: {check_memory_usage():.2f} MB")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    train_ds = MessageDataset(train_df, tokenizer)
    val_ds   = MessageDataset(val_df,   tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=2e-5,

        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=1,
        dataloader_num_workers=1,
        fp16=False,
        optim='adamw_torch'
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("🚀 학습 시작...")
    trainer.train()
    print("✅ 학습 완료!")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"📦 저장 완료: {output_dir}")

# 🧪 모델 평가 함수
def evaluate_model(test_file, model_dir):
    df = load_data(test_file)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tok   = AutoTokenizer.from_pretrained(model_dir)
    correct = 0
    for i in range(0, len(df), 4):
        batch = df.iloc[i:i+4]
        for _, row in batch.iterrows():
            prem = f"'{row['relationship']}' 관계에서 메시지가 적합한가요?"
            hyp  = row['message']
            enc  = tok(prem, hyp, return_tensors="pt", max_length=128,
                       truncation=True, padding='max_length')
            with torch.no_grad():
                out = model(**enc)
            pred = torch.argmax(out.logits, dim=-1).item()
            if pred == row['label']:
                correct += 1
    acc = correct/len(df)*100
    print(f"📊 평가 정확도: {acc:.2f}%")
    return acc

# 전역 변수 선언 (inference)
model = None
tokenizer = None

# 모델 로드 (inference)
def load_model(model_dir):
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return True

# 예측 함수 (inference)
def predict(text: str, relationship: str = "일반"):
    if model is None or tokenizer is None:
        raise ValueError("모델이 로드되지 않았습니다.")
    prem = f"'{relationship}' 관계에서 메시지가 적합한가요?"
    enc  = tokenizer(prem, text,
                     truncation=True, padding=True,
                     return_tensors="pt", max_length=128)
    with torch.no_grad():
        out = model(**enc)
    logits     = out.logits
    pred_idx   = torch.argmax(logits, dim=-1).item()
    probs      = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][pred_idx].item() * 100
    return pred_idx, confidence

# 결과 해석 (inference)
def interpret_result(pred_idx: int) -> str:
    return {0: "부적절 표현", 1: "중립 표현", 2: "적절 표현"}.get(pred_idx, "알 수 없음")
