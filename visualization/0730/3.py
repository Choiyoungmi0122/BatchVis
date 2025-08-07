from sentence_transformers import SentenceTransformer
import umap
import plotly.express as px
import pandas as pd
import json

# 모델 로드
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# JSON 파일 로드
file_path = "C:/Users/Young-Mi Choi/Desktop/dev/batchpro\myapi/responses/experiment_20250724_154245.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data["history"]

# 특정 질문
target_question = "어디가 불편해서 방문하게 되었나요?"

# 데이터 수집
answers = []
labels = []
tci_list = []

for i, record in enumerate(records):
    sim_label = f"Sim {i+1}"
    qa_list = record["prompts"][0]["qa"]
    tci_detail = record["prompts"][0]["detail"]
    temperament = tci_detail["temperament"]
    character = tci_detail["character"]
    tci_str = f"{temperament} | {character}"

    for qa in qa_list:
        if qa["question"].strip() == target_question:
            answer = qa["answer"].strip()
            answers.append(answer)
            labels.append(sim_label)
            tci_list.append(tci_str)

# 임베딩
embeddings = model.encode(answers)

# UMAP 차원 축소
reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# 시각화용 데이터프레임 구성
df = pd.DataFrame(embedding_2d, columns=["x", "y"])
df["label"] = labels
df["answer"] = answers
df["TCI"] = tci_list

# Plotly 시각화
fig = px.scatter(
    df,
    x="x", y="y",
    text="label",
    hover_data={"answer": True, "TCI": True, "label": False},
    title="동일 질문에 대한 시뮬레이션 응답 분포 (UMAP)"
)
fig.update_traces(textposition='top center')
fig.update_layout(height=600, width=800)
fig.show()
