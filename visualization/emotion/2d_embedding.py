from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import numpy as np
import json

# ---------------- 설정 ---------------- #
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"      # 워드클라우드용 한글 폰트 경로
FILE_PATH = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json"
# -------------------------------------- #


# 모델 로드
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# JSON 파일 로드
file_path = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["history"]

# 각 시뮬레이션별 전체 답변을 하나의 텍스트로 묶음
all_answers = [" ".join([qa["answer"] for qa in record["prompts"][0]["qa"]]) for record in records]
labels = [f"Sim {i+1}" for i in range(len(records))]

# 문장 임베딩
embeddings = model.encode(all_answers)

# UMAP 차원 축소
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# 시각화 (출력 주석 처리됨)
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    x, y = embedding_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, label, fontsize=9)
plt.title("시뮬레이션 응답의 2D 임베딩 분포 (UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()
