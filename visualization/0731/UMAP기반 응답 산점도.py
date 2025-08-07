import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import umap.umap_ as umap

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

# 1. 데이터 로드
with open('C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompts = data['history'][0]['prompts']
responses = [item['qa'][0]['answer'] for item in prompts]
mapping = {'하': 1, '중': 2, '상': 3}
autonomy_levels = [mapping[item['detail']['character']['자율성']] for item in prompts]

# 2. TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(responses)

# 3. UMAP 차원 축소
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=10,    # 근접 이웃 수
    min_dist=0.1       # 저차원 상의 최소 거리
)
X_umap = reducer.fit_transform(X.toarray())

# 4. 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=autonomy_levels,
    cmap='viridis',
    s=50,
    alpha=0.8
)
cbar = plt.colorbar(scatter, label='자율성 레벨 (1=하, 2=중, 3=상)')
plt.title('UMAP 기반 페르소나 응답 산점도')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.tight_layout()
plt.show()
