# myapi - BatchPro 백엔드 API

## 개요
이 폴더는 BatchPro 시스템의 핵심 백엔드 API를 포함합니다. FastAPI 기반으로 구축되어 가상환자 페르소나 생성, 분석, 시각화 기능을 제공합니다.

## 주요 파일

### 🚀 **핵심 API 파일**
- **`main.py`** - 메인 FastAPI 애플리케이션 (25개 TCI 조합 지원)
- **`run_server.py`** - 서버 실행 스크립트
- **`visualization.py`** - 시각화 및 분석 API 엔드포인트

### 🔧 **설정 및 유틸리티**
- **`batch_config.py`** - OpenAI Batch API 설정
- **`experiment.py`** - 실험 관리 및 데이터 처리
- **`analysis.py`** - 데이터 분석 유틸리티

### 📊 **테스트 및 성능**
- **`test_api.py`** - API 통합 테스트
- **`test_batch_processing.py`** - Batch API 처리 테스트
- **`performance_comparison.py`** - 성능 비교 분석
- **`test_storage.py`** - 저장 시스템 테스트

### 📁 **데이터 폴더**
- **`responses/`** - 실험 응답 데이터 저장
- **`saved_analyses/`** - 분석 결과 자동 저장
- **`frontend/`** - Svelte 기반 프론트엔드

## API 구조

### **기본 엔드포인트**
- `POST /start_experiment` - 실험 번호 생성
- `POST /process_qa_batch` - Batch API를 통한 가상환자 응답 생성
- `GET /list_experiments` - 저장된 실험 목록 조회

### **시각화 엔드포인트**
- `POST /visualization/generate` - 기본 시각화 생성
- `POST /visualization/structured-analysis` - 구조화된 분석
- `GET /visualization/saved-analyses` - 저장된 분석 목록

### **Batch API 관리**
- `POST /start_batch` - Batch 작업 시작
- `GET /check_batch_status/{batch_id}` - Batch 상태 확인
- `GET /download_batch_results/{batch_id}` - 결과 다운로드

## 실행 방법

### **1. 환경 설정**
```bash
conda activate batchpro
cd myapi
```

### **2. 서버 실행**
```bash
# 방법 1: run_server.py 사용
python run_server.py

# 방법 2: uvicorn 직접 사용
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **3. API 문서 접속**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 주요 기능

### **TCI 조합 처리**
- **25개 선택된 조합** 사용으로 처리 시간 최적화
- **기질(Temperament)**: 자극추구, 위험회피, 사회성, 인내성
- **성격(Character)**: 자율성, 연대감, 자기초월성

### **Batch API 통합**
- OpenAI Batch API를 통한 대량 처리
- **275개 응답** (25개 조합 × 11개 질문) 처리
- 예상 처리 시간: **27-28분**

### **자동 저장 시스템**
- 분석 결과 자동 저장 및 재사용
- 해시 기반 중복 감지
- 메타데이터 분리 저장

## 개발 가이드

### **새로운 API 추가**
1. `main.py`에 새로운 엔드포인트 정의
2. 필요한 데이터 모델을 `visualization.py`에 추가
3. 테스트 코드 작성
4. API 문서 업데이트

### **성능 최적화**
- Batch 처리 활용
- 캐싱 전략 구현
- 비동기 처리 적용

## 문제 해결

### **일반적인 오류**
- **ModuleNotFoundError**: `conda activate batchpro` 실행
- **API 키 오류**: `.env` 파일에 `OPENAI_API_KEY` 설정
- **포트 충돌**: 다른 포트 사용 또는 기존 프로세스 종료

### **Batch API 오류**
- **Rate Limit**: API 호출 간격 조정
- **네트워크 오류**: 연결 상태 및 API 키 확인
- **메모리 부족**: 배치 크기 조정

## 의존성

### **필수 패키지**
```bash
pip install fastapi uvicorn openai python-dotenv
pip install sentence-transformers scikit-learn numpy pandas
```

### **환경 변수**
```bash
OPENAI_API_KEY=your_api_key_here
```

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다.
