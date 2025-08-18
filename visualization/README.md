# visualization - 시각화 및 분석 스크립트

## 개요
이 폴더는 BatchPro 시스템의 시각화 및 데이터 분석을 위한 스크립트들과 결과물들을 포함합니다. 다양한 차트, 그래프, 분석 결과를 생성하고 저장합니다.

## 폴더 구조

### 📊 **메인 시각화**
- **`visualization_index.html`** - 통합 시각화 대시보드
  - 레이더 차트, 히트맵, 정렬 차트
  - 페르소나별 응답 비교
  - 실시간 데이터 필터링

### 📅 **날짜별 분석 스크립트**
- **`0730/`** - 7월 30일 분석 스크립트
  - `3.py` - 3차원 시각화
  - `4.py` - 4차원 분석
  - `5.py` - 5차원 데이터 처리
  - `tci_chart.py` - TCI 차트 생성

- **`0731/`** - 7월 31일 분석 스크립트
  - `cluster keywords.py` - 키워드 클러스터링
  - `t-SNE 기반 페르소나 응답 산점도.py` - t-SNE 차원 축소
  - `UMAP기반 응답 산점도.py` - UMAP 차원 축소
  - `군집화.py` - K-means 클러스터링

- **`0813/`** - 8월 13일 분석 스크립트
  - 새로운 분석 방법론 및 실험

### 😊 **감정 분석**
- **`emotion/`** - 감정 분석 전용 스크립트
  - `1.py` - 기본 감정 분석
  - `2d_embedding.py` - 2차원 임베딩 시각화
  - `emotion_analysis_full.py` - 전체 감정 분석 파이프라인

## 주요 기능

### **차트 및 그래프**
1. **레이더 차트**: TCI 성향별 다차원 프로파일
2. **히트맵**: 페르소나 간 응답 유사도
3. **산점도**: t-SNE, UMAP 기반 차원 축소
4. **클러스터링**: K-means, DBSCAN 군집화
5. **감정 분석**: 텍스트 기반 감정 강도 및 방향

### **분석 방법론**
- **차원 축소**: 고차원 데이터를 2D로 시각화
- **클러스터링**: 유사한 응답 패턴 그룹화
- **키워드 분석**: 응답 텍스트의 핵심 단어 추출
- **감정 분석**: 긍정/부정/중립 감정 분류

## 사용 방법

### **1. 기본 시각화**
```bash
# 메인 시각화 대시보드 열기
open visualization_index.html
```

### **2. Python 스크립트 실행**
```bash
# conda 환경 활성화
conda activate batchpro

# 특정 분석 스크립트 실행
cd visualization/0731
python "t-SNE 기반 페르소나 응답 산점도.py"
```

### **3. 감정 분석 실행**
```bash
cd visualization/emotion
python emotion_analysis_full.py
```

## 데이터 요구사항

### **입력 데이터 형식**
- **JSON 파일**: 실험 결과 데이터
- **응답 텍스트**: 페르소나별 질문 응답
- **메타데이터**: TCI 성향 정보, 실험 설정

### **출력 결과**
- **차트 이미지**: PNG, JPG, SVG 형식
- **분석 데이터**: CSV, JSON 형식
- **인터랙티브 차트**: HTML 기반 대시보드

## 기술 스택

### **Python 라이브러리**
- **시각화**: matplotlib, seaborn, plotly
- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn
- **차원 축소**: t-SNE, UMAP
- **클러스터링**: K-means, DBSCAN

### **웹 기술**
- **HTML/CSS/JavaScript**: 기본 웹 인터페이스
- **Chart.js**: 인터랙티브 차트
- **D3.js**: 고급 데이터 시각화

## 개발 가이드

### **새로운 시각화 추가**
1. Python 스크립트 작성
2. 필요한 라이브러리 import
3. 데이터 로드 및 전처리
4. 시각화 생성 및 저장
5. README.md 업데이트

### **성능 최적화**
- **대용량 데이터**: 청크 단위 처리
- **메모리 관리**: 불필요한 데이터 제거
- **병렬 처리**: multiprocessing 활용

## 문제 해결

### **일반적인 오류**
- **ModuleNotFoundError**: 필요한 라이브러리 설치
- **메모리 부족**: 데이터 크기 줄이기
- **차트 렌더링 오류**: matplotlib 백엔드 설정

### **데이터 문제**
- **형식 불일치**: JSON 스키마 확인
- **누락된 데이터**: 전처리 단계에서 처리
- **인코딩 문제**: UTF-8 인코딩 확인

## 예제 코드

### **기본 시각화 생성**
```python
import matplotlib.pyplot as plt
import pandas as pd

# 데이터 로드
data = pd.read_json('experiment_data.json')

# 시각화 생성
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'])
plt.title('페르소나 응답 분포')
plt.savefig('scatter_plot.png')
plt.close()
```

### **t-SNE 차원 축소**
```python
from sklearn.manifold import TSNE

# t-SNE 적용
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 시각화
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title('t-SNE 기반 페르소나 응답 분포')
```

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
