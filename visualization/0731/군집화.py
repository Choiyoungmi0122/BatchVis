import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

# Load the data
with open('C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = data['history'][0]['prompts']
responses = [item['qa'][0]['answer'] for item in prompts]

# 2. TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(responses)

# 3. 차원 축소 (UMAP 우선, 설치되지 않은 경우 t-SNE 대체)
try:
    import umap.umap_ as umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10, min_dist=0.1)
    embedding = reducer.fit_transform(X.toarray())
    method = 'UMAP'
except ImportError:
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    embedding = reducer.fit_transform(X.toarray())
    method = 't-SNE'

# 4. KMeans 군집화
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embedding)

# 5. 시각화
plt.figure(figsize=(8, 6))
markers = ['o', 's', '^']
for i in range(n_clusters):
    idx = labels == i
    plt.scatter(embedding[idx, 0], embedding[idx, 1], marker=markers[i], label=f'Cluster {i}')
plt.legend()
plt.title(f'{method} 기반 페르소나 응답 {n_clusters}개 군집화')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.tight_layout()
plt.show()
