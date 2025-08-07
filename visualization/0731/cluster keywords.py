import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import ace_tools_open as tools

# 1. 데이터 로드
with open('C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = data['history'][0]['prompts']
responses = [item['qa'][0]['answer'] for item in prompts]

# 2. TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(responses)

# 3. KMeans 군집화 (3개)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

# 4. 군집별 키워드 추출 및 대표 텍스트 선택
feature_names = vectorizer.get_feature_names_out()
cluster_data = []

for i in range(n_clusters):
    idx = np.where(labels == i)[0]
    X_cluster = X[idx]
    
    # 키워드: 군집 중심에서 상위 10개 피처
    centroid = kmeans.cluster_centers_[i]
    top_indices = centroid.argsort()[::-1][:10]
    keywords = [feature_names[j] for j in top_indices]
    
    # 대표 텍스트: 중심에 가장 가까운 응답
    cluster_vectors = X_cluster.toarray()
    distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
    rep_idx = idx[np.argmin(distances)]
    representative_text = responses[rep_idx]
    
    cluster_data.append({
        'cluster': f'Cluster {i}',
        'keywords': ', '.join(keywords),
        'representative_text': representative_text
    })

df = pd.DataFrame(cluster_data)

# 5. 결과 표시
tools.display_dataframe_to_user("Cluster Keywords and Representative Texts", df)
