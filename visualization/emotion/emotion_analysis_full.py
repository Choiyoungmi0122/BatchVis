# -*- coding: utf-8 -*-
import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 설정 ---------------- #
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"      # 워드클라우드용 한글 폰트 경로
FILE_PATH = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json"
# -------------------------------------- #

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

# JSON 로드
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["history"]

# 질문 리스트 추출
questions = [qa["question"] for qa in records[0]["prompts"][0]["qa"]]
labels = [f"Sim {i+1}" for i in range(len(records))]

# 감정 매트릭스 초기화
emotion_matrices = {
    emotion: pd.DataFrame(index=labels, columns=questions)
    for emotion in EMOTIONS
}

# 감정 분석 저장
for i, record in enumerate(records):
    qa_list = record["prompts"][0]["qa"]
    for qa in qa_list:
        question = qa["question"]
        answer = qa["answer"]
        scores = analyze_emotion(answer)
        for emotion in EMOTIONS:
            emotion_matrices[emotion].loc[f"Sim {i+1}", question] = scores[emotion]

# 타입 변환
for emotion in EMOTIONS:
    emotion_matrices[emotion] = emotion_matrices[emotion].astype(float)

# ---------------- 히트맵 시각화 ---------------- #
# for emotion, df in emotion_matrices.items():
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(df, annot=True, cmap="Reds", cbar=True)
#     plt.title(f"{emotion} 감정 히트맵")
#     plt.ylabel("가상환자")
#     plt.xlabel("질문")
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.show()

# ---------------- 유사도 히트맵 ---------------- #
# all_answers = [" ".join([qa["answer"] for qa in record["prompts"][0]["qa"]]) for record in records]
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(all_answers)
# sim_matrix = cosine_similarity(X)

# plt.figure(figsize=(8, 6))
# sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Blues", fmt=".2f")
# plt.title("시뮬레이션 간 텍스트 유사도 (Cosine Similarity)")
# plt.tight_layout()
# plt.show()

# ---------------- 워드클라우드 ---------------- #
# for i, record in enumerate(records):
#     text = " ".join([qa["answer"] for qa in record["prompts"][0]["qa"]])
#     wordcloud = WordCloud(font_path=FONT_PATH, background_color="white", width=800, height=600).generate(text)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     ## plt.title(f"Sim {i+1} 워드클라우드")
#     plt.tight_layout()
#     plt.show()

# ---------------- 감정 평균 바차트 ---------------- #
# summary_df = pd.DataFrame(index=EMOTIONS.keys(), columns=["mean", "std"])
# for emotion in EMOTIONS:
#     df = emotion_matrices[emotion]
#     summary_df.loc[emotion, "mean"] = df.values.mean()
#     summary_df.loc[emotion, "std"] = df.values.std()
# summary_df = summary_df.astype(float)

# plt.figure(figsize=(8, 5))
# summary_df["mean"].plot(kind="bar", yerr=summary_df["std"], capsize=5, color="salmon")
# plt.ylabel("감정 점수 평균 (±표준편차)")
# plt.title("감정별 평균 점수 및 분산")
# plt.tight_layout()
# plt.show()
