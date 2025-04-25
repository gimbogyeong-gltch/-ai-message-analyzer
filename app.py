
import tkinter as tk
from tkinter import messagebox, filedialog
from model_utils import load_model, predict, interpret_result
import threading


# 개선된 GUI 클래스
class MessageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 메시지 분석")
        self.root.geometry("600x500")
        self.root.configure(bg="#f0f0f0")
        
        # 모델 로드 상태
        self.model_loaded = False
        self.model_dir = ""
        
        # 메인 프레임
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 타이틀
        title_label = tk.Label(
            self.main_frame, 
            text="AI 메시지 분석기", 
            font=("Helvetica", 18, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=10)
        
        # 모델 로드 프레임
        model_frame = tk.LabelFrame(
            self.main_frame, 
            text="모델 로드", 
            bg="#f0f0f0",
            font=("Helvetica", 10)
        )
        model_frame.pack(fill=tk.X, pady=10)
        
        self.model_path_var = tk.StringVar()
        model_path_entry = tk.Entry(
            model_frame, 
            textvariable=self.model_path_var,
            width=50
        )
        model_path_entry.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
        
        browse_button = tk.Button(
            model_frame, 
            text="찾아보기", 
            command=self.browse_model
        )
        browse_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        load_button = tk.Button(
            model_frame, 
            text="모델 로드", 
            command=self.load_model_action
        )
        load_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        # 입력 프레임
        input_frame = tk.LabelFrame(
            self.main_frame, 
            text="메시지 입력", 
            bg="#f0f0f0",
            font=("Helvetica", 10)
        )
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 관계 선택
        relationship_frame = tk.Frame(input_frame, bg="#f0f0f0")
        relationship_frame.pack(fill=tk.X, padx=10, pady=5)
        
        relationship_label = tk.Label(
            relationship_frame, 
            text="관계:", 
            bg="#f0f0f0",
            width=10,
            anchor="w"
        )
        relationship_label.pack(side=tk.LEFT, padx=5)
        
        self.relationship_var = tk.StringVar(value="일반")
        relationship_options = ["일반", "친구", "연인", "가족", "직장", "학교"]
        relationship_dropdown = tk.OptionMenu(
            relationship_frame, 
            self.relationship_var, 
            *relationship_options
        )
        relationship_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 메시지 입력
        message_frame = tk.Frame(input_frame, bg="#f0f0f0")
        message_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        message_label = tk.Label(
            message_frame, 
            text="메시지:", 
            bg="#f0f0f0",
            width=10,
            anchor="w"
        )
        message_label.pack(side=tk.LEFT, padx=5, anchor="n")
        
        self.message_text = tk.Text(
            message_frame, 
            height=5,
            wrap=tk.WORD
        )
        self.message_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
        
        # 분석 버튼
        analyze_button = tk.Button(
            input_frame, 
            text="메시지 분석하기", 
            command=self.analyze_message,
            height=2
        )
        analyze_button.pack(pady=10)
        
        # 결과 프레임
        result_frame = tk.LabelFrame(
            self.main_frame, 
            text="분석 결과", 
            bg="#f0f0f0",
            font=("Helvetica", 10)
        )
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = tk.Label(
            result_frame, 
            text="아직 분석되지 않았습니다.",
            font=("Helvetica", 12),
            bg="#f0f0f0",
            pady=15
        )
        self.result_label.pack()
        
        # 상태바
        self.status_var = tk.StringVar(value="준비 완료")
        status_bar = tk.Label(
            root, 
            textvariable=self.status_var, 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def browse_model(self):
        folder_path = filedialog.askdirectory(title="모델 폴더 선택")
        if folder_path:
            self.model_path_var.set(folder_path)
    
    def load_model_action(self):
        model_dir = self.model_path_var.get()
        if not model_dir:
            messagebox.showerror("오류", "모델 경로를 입력하세요.")
            return
        
        self.status_var.set("모델 로딩 중...")
        self.root.update_idletasks()
        
        # 별도 스레드에서 모델 로드
        def load_model_thread():
            success = load_model(model_dir)
            if success:
                self.model_loaded = True
                self.model_dir = model_dir
                self.status_var.set("모델 로드 완료")
                messagebox.showinfo("성공", "모델 로드 완료!")
            else:
                self.status_var.set("모델 로드 실패")
                messagebox.showerror("오류", "모델을 로드할 수 없습니다.")
        
        threading.Thread(target=load_model_thread).start()
    
    def analyze_message(self):
        if not self.model_loaded:
            messagebox.showerror("오류", "먼저 모델을 로드하세요.")
            return

        raw = self.message_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showerror("오류", "메시지를 입력하세요.")
            return

        relationship = self.relationship_var.get()

        self.status_var.set("분석 중...")
        self.root.update_idletasks()

        try:
            # 실제 예측: (레이블 인덱스, 신뢰도)
            pred_idx, confidence = predict(raw, relationship)

            # 0·1·2 를 사람이 읽을 문자열로 변환
            label_str = interpret_result(pred_idx)  


            # 결과 표시
            self.result_label.config(
                text=f"예측 결과: {label_str} (신뢰도: {confidence:.1f}%)",
                fg=("green" if pred_idx == 2 else
                    "black" if pred_idx == 1 else
                    "red")
            )
            self.status_var.set("분석 완료")

        except Exception as e:
            self.status_var.set("분석 오류")
            messagebox.showerror("오류", f"분석 중 오류 발생: {e}")


# 메인 함수
def main():
    root = tk.Tk()
    app = MessageAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
