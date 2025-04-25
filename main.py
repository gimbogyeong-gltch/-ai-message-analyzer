from analyzer import ToneAnalyzer
from detector import OffensiveDetector
from database import ClosenessDatabase
from decision import DecisionMaker


class MessageProcessor:
    def __init__(self):
        self.tone_analyzer = ToneAnalyzer()
        self.detector = OffensiveDetector()
        self.db = ClosenessDatabase()
        self.decision_maker = DecisionMaker()

    def process(self, user_id: str, target_id: str, message: str) -> dict:
        tone = self.tone_analyzer.predict(message)
        offense = self.detector.predict(message)
        closeness = self.db.get_closeness(user_id, target_id)
        decision = self.decision_maker.evaluate(tone, offense, closeness)

        return {
            "입력 메시지": message,
            "말투 분석": tone,
            "욕설 여부": offense,
            "친밀도": closeness,
            "최종 판단": decision
        }


if __name__ == "__main__":
    processor = MessageProcessor()

    user_id = input("👤 사용자 ID 입력: ")
    target_id = input("👤 대상 ID 입력: ")
    message = input("💬 메시지 입력: ")

    result = processor.process(user_id, target_id, message)

    print("\n📊 분석 결과:")
    for key, value in result.items():
        print(f"{key}: {value}")
