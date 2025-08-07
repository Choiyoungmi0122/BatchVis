import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------- 설정 ---------------- #
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"      # 워드클라우드용 한글 폰트 경로
# -------------------------------------- #

# JSON 파일 로드
file_path = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro\myapi/responses/experiment_20250724_154245.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["history"]
target_question = "어디가 불편해서 방문하게 되었나요?"

# 답변 수집
answers = []
labels = []

for i, record in enumerate(records):
    label = f"Sim {i+1}"
    qa_list = record["prompts"][0]["qa"]
    for qa in qa_list:
        if qa["question"].strip() == target_question:
            answers.append(qa["answer"].strip())
            labels.append(label)

# 1. n-gram 분석 (bi-gram)
vectorizer_ngram = CountVectorizer(ngram_range=(2, 2), stop_words=["저", "좀", "것", "너무", "이런", "그런", "그리고", "있어요", "합니다", "해서", "있습니다", "그게", "때문에"])
X_ngram = vectorizer_ngram.fit_transform(answers)
ngram_freq = X_ngram.sum(axis=0)
ngram_df = pd.DataFrame({
    "ngram": vectorizer_ngram.get_feature_names_out(),
    "count": ngram_freq.A1
}).sort_values(by="count", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x="count", y="ngram", data=ngram_df, palette="Greens_d")
plt.title("상위 20개 bi-gram 표현")
plt.xlabel("빈도수")
plt.ylabel("표현")
plt.tight_layout()
plt.show()

# 2. KMeans 클러스터링 (TF-IDF 기반 + PCA 시각화)
vectorizer_tfidf = TfidfVectorizer(stop_words=["저", "좀", "것", "너무", "이런", "그런", "그리고", "있어요", "합니다", "해서", "있습니다", "그게", "때문에"])
X_tfidf = vectorizer_tfidf.fit_transform(answers)

# KMeans 클러스터링
k = 4  # 클러스터 수
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_tfidf)

# PCA 시각화
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_tfidf.toarray())
cluster_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
cluster_df["Cluster"] = clusters
cluster_df["Sim"] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="Cluster", style="Cluster", s=100)
for i, row in cluster_df.iterrows():
    plt.text(row["PC1"] + 0.01, row["PC2"] + 0.01, row["Sim"], fontsize=9)
plt.title("KMeans 기반 응답 클러스터링 (PCA 시각화)")
plt.tight_layout()
plt.show()