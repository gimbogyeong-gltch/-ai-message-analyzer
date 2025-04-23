# analyzer.py

class ToneAnalyzer:
    """
    말투(어조)를 분석하는 클래스.
    나중에 GPT 또는 KoBERT 기반 모델로 교체 가능.
    """

    def predict(self, text: str) -> str:
        if "야" in text or "뭐" in text or "왜" in text:
            return "공격적"
        return "중립"
# analyzer.py

class ToneAnalyzer:
    """
    말투를 분석하는 클래스.
    추후 ChatGPT 또는 KoBERT 연동 가능.
    지금은 임시 키워드 기반 분석.
    """

    def predict(self, text: str) -> str:
        if "야" in text or "뭐" in text or "왜" in text:
            return "공격적"
        return "중립"
