# run_train.py

import argparse
from model_utils import train_model

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI 메시지 분석 모델 학습 스크립트"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="학습용 CSV 파일 경로 (e.g. my_data.csv)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="snunlp/KR-FinBERT",
        help="사전 학습 모델 이름 (HuggingFace 허브)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="학습 배치 크기"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="학습 에폭 수"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_output",
        help="학습된 모델 저장 폴더"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    train_model(
        train_file   = args.train_file,
        model_name   = args.model_name,
        batch_size   = args.batch_size,
        epochs       = args.epochs,
        output_dir   = args.output_dir,
    )

if __name__ == "__main__":
    main()
