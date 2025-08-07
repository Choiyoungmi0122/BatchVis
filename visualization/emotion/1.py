import json
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (로컬 실행 환경에 맞게)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기준

# 감정 키워드 정의
EMOTIONS = {
    "우울": ["우울", "무기력", "가라앉", "의욕", "무의미", "절망", "지치", "힘들", "포기"],
    "불안": ["불안", "걱정", "두렵", "겁", "초조", "긴장", "망설"],
    "분노": ["짜증", "화", "분노", "성질", "예민", "폭발", "신경질"],
}

# 감정 분석 함수
def analyze_emotion(text):
    scores = defaultdict(int)
    for emotion, keywords in EMOTIONS.items():
        for word in keywords:
            scores[emotion] += len(re.findall(word, text))
    return scores

# JSON 파일 경로
file_path = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json"

# JSON 로드
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["history"]

# 질문 목록 추출 (첫 번째 시뮬레이션 기준)
questions = []
for prompt in records[0]["prompts"]:
    for qa in prompt["qa"]:
        questions.append(qa["question"])
    break

# 감정별 히트맵을 위한 데이터프레임 초기화
emotion_matrices = {
    emotion: pd.DataFrame(index=[f"Sim {i+1}" for i in range(len(records))], columns=questions)
    for emotion in EMOTIONS.keys()
}

# 각 시뮬레이션 별로 모든 답변 분석하여 점수 기록
for i, record in enumerate(records):
    qa_list = record["prompts"][0]["qa"]
    for qa in qa_list:
        question = qa["question"]
        answer = qa["answer"]
        scores = analyze_emotion(answer)
        for emotion in EMOTIONS:
            emotion_matrices[emotion].loc[f"Sim {i+1}", question] = scores[emotion]

# 데이터 타입 float으로 전환
for emotion in EMOTIONS:
    emotion_matrices[emotion] = emotion_matrices[emotion].astype(float)

# 히트맵 시각화 (실행은 나중에)
# for emotion, df in emotion_matrices.items():
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(df, annot=True, cmap="Reds", cbar=True)
#     plt.title(f"{emotion} 감정 히트맵")
#     plt.ylabel("가상환자")
#     plt.xlabel("질문")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()
