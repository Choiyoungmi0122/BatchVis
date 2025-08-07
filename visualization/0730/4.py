from wordcloud import WordCloud
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# ---------------- 설정 ---------------- #
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"      # 워드클라우드용 한글 폰트 경로
FILE_PATH = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json"
# -------------------------------------- #

# JSON 파일 로드
file_path = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro\myapi/responses/experiment_20250724_154245.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["history"]
target_question = "어디가 불편해서 방문하게 되었나요?"

answers = []
labels = []

for i, record in enumerate(records):
    label = f"Sim {i+1}"
    qa_list = record["prompts"][0]["qa"]
    for qa in qa_list:
        if qa["question"].strip() == target_question:
            answers.append(qa["answer"].strip())
            labels.append(label)

# 워드클라우드
full_text = " ".join(answers)
wordcloud = WordCloud(background_color="white", width=800, height=600).generate(full_text)

plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("워드클라우드: 전체 응답 단어 분포")
plt.tight_layout()
plt.show()

# 상위 키워드 bar chart
vectorizer = CountVectorizer(stop_words=["저", "좀", "것", "너무", "이런", "그런", "그리고", "있어요", "합니다", "해서", "있습니다", "그게", "때문에"])
X = vectorizer.fit_transform(answers)
word_freq = X.sum(axis=0)
word_freq_df = pd.DataFrame({
    "word": vectorizer.get_feature_names_out(),
    "count": word_freq.A1
}).sort_values(by="count", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x="count", y="word", data=word_freq_df, palette="Blues_d")
plt.title("상위 20개 키워드")
plt.xlabel("빈도수")
plt.ylabel("단어")
plt.tight_layout()
plt.show()

# 단어-시뮬레이션 히트맵
top_words = word_freq_df["word"].tolist()[:15]
vectorizer_top = CountVectorizer(vocabulary=top_words)
X_top = vectorizer_top.fit_transform(answers)
heatmap_df = pd.DataFrame(X_top.toarray(), index=labels, columns=top_words)

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df, annot=True, cmap="Oranges", fmt="d")
plt.title("단어 사용 히트맵 (시뮬레이션별 상위 키워드)")
plt.xlabel("단어")
plt.ylabel("시뮬레이션")
plt.tight_layout()
plt.show()
