import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

# Load the data
with open('C:/Users/Young-Mi Choi/Desktop/dev/batchpro/myapi/responses/experiment_20250724_154245.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract responses and autonomy levels
prompts = data['history'][0]['prompts']
responses = [item['qa'][0]['answer'] for item in prompts]
mapping = {'하': 1, '중': 2, '상': 3}
autonomy_levels = [mapping[item['detail']['character']['자율성']] for item in prompts]

# Vectorize responses
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(responses)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
X_tsne = tsne.fit_transform(X.toarray())

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=autonomy_levels)
plt.colorbar(scatter, label='자율성 레벨 (1=하, 2=중, 3=상)')
plt.title('t-SNE 기반 페르소나 응답 산점도')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.show()
