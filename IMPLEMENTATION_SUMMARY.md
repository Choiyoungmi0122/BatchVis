# 자동 분석 결과 로딩 시스템 구현 요약

## 🎯 구현 목표
사용자의 요청: "저장된 분석 결과를 따로 만들지 말고, 해당 환자의 해당 질문에 대해서 이미 한게 있으면 그대로 불러오게 해줘 히트맵이나 다 마찬가지로"

## 🔄 주요 변경사항

### 1. AnalysisStorage 클래스 수정

#### 변경 전:
- 각 분석 타입별로 별도의 파일 저장
- `_generate_analysis_id(experiment_name, question_index, analysis_type)` - 분석 타입 포함
- `save_analysis(experiment_name, question_index, analysis_data, analysis_type)` - 분석 타입별 저장

#### 변경 후:
- **단일 통합 분석 파일**로 저장 (실험명 + 질문 인덱스 기반)
- `_generate_analysis_id(experiment_name, question_index)` - 분석 타입 제거
- `save_analysis(experiment_name, question_index, analysis_data)` - 분석 타입 제거
- **기존 분석 덮어쓰기** 기능 추가

#### 새로운 메서드:
```python
def _delete_old_versions(self, experiment_name: str, question_index: int):
    """같은 experiment_name, question_index의 기존 분석 파일들 삭제"""

def find_analysis_by_params(self, experiment_name: str, question_index: int) -> Optional[str]:
    """실험명과 질문 인덱스로 기존 분석 찾기"""
```

### 2. 시각화 생성 엔드포인트 (`/visualization/generate`) 수정

#### 새로운 로직:
1. **기존 분석 확인**: `analysis_storage.find_analysis_by_params()` 호출
2. **자동 로딩**: 기존 분석이 있으면 즉시 로드하여 반환
3. **스마트 필터링**: 요청된 분석 타입만 필터링하여 반환
4. **캐시 상태 표시**: `loaded_from_cache: true/false` 응답에 포함

#### 응답 형식:
```json
{
    "success": true,
    "data": {...},
    "message": "기존 분석 결과를 로드했습니다",
    "analysis_id": "experiment_1_Q0",
    "loaded_from_cache": true
}
```

### 3. 구조화된 분석 엔드포인트 (`/visualization/structured-analysis`) 수정

#### 새로운 로직:
1. **기존 분석 확인**: 동일한 실험/질문에 대한 기존 분석 검색
2. **기존 분석 업데이트**: 기존 분석에 `structured_analysis` 추가
3. **통합 저장**: 업데이트된 전체 분석을 단일 파일로 저장

### 4. 프론트엔드 업데이트

#### 새로운 기능:
- **알림 시스템**: `showNotification()` 함수로 캐시 로딩/새 생성 상태 표시
- **자동 새로고침**: 분석 완료 후 저장된 분석 목록 자동 업데이트
- **사용자 피드백**: "기존 분석 결과를 로드했습니다" vs "새로운 분석을 생성했습니다"

#### 알림 타입:
- 🔵 **정보**: 기존 분석 로드
- 🟢 **성공**: 새로운 분석 생성
- 🟡 **경고**: 주의사항
- 🔴 **오류**: 오류 발생

## 🗂️ 저장 구조 변경

### 변경 전:
```
saved_analyses/
├── experiment_1_Q0_radar_abc123.pkl
├── experiment_1_Q0_radar_abc123_metadata.json
├── experiment_1_Q0_heatmap_def456.pkl
├── experiment_1_Q0_heatmap_def456_metadata.json
├── experiment_1_Q0_structured_ghi789.pkl
└── experiment_1_Q0_structured_ghi789_metadata.json
```

### 변경 후:
```
saved_analyses/
├── experiment_1_Q0_xyz123.pkl          # 통합 분석 데이터
└── experiment_1_Q0_xyz123_metadata.json # 통합 메타데이터
```

## 🚀 사용자 경험 개선

### 1. **즉시 응답**
- 기존 분석이 있으면 재생성 없이 즉시 로드
- 대기 시간 대폭 단축

### 2. **자동 관리**
- 같은 실험/질문에 대한 분석은 자동으로 덮어쓰기
- 중복 파일 없이 깔끔한 저장 구조

### 3. **투명한 상태 표시**
- 사용자가 언제 캐시에서 로드되고 언제 새로 생성되는지 명확히 알 수 있음
- 알림 시스템으로 실시간 피드백 제공

## 🔧 기술적 장점

### 1. **효율성**
- 불필요한 재계산 방지
- 저장 공간 절약
- 응답 속도 향상

### 2. **일관성**
- 단일 분석 ID로 모든 시각화 타입 관리
- 데이터 무결성 보장

### 3. **확장성**
- 새로운 시각화 타입 추가 시 기존 분석과 통합 가능
- 분석 메타데이터 중앙 집중 관리

## 📋 테스트 방법

### 1. **기본 기능 테스트**
```bash
python test_storage_updated.py
```

### 2. **웹 인터페이스 테스트**
1. 백엔드 서버 실행: `cd myapi && python run_server.py`
2. 브라우저에서 `http://127.0.0.1:8000/visualization` 접속
3. 같은 실험/질문으로 분석 실행
4. 두 번째 실행 시 "기존 분석 결과를 로드했습니다" 알림 확인

## ⚠️ 주의사항

### 1. **데이터 덮어쓰기**
- 같은 실험/질문에 대한 새로운 분석은 기존 분석을 완전히 덮어씀
- 기존 분석 보존이 필요한 경우 별도 백업 필요

### 2. **호환성**
- 기존 저장된 분석 파일들은 새로운 시스템과 호환되지 않음
- 시스템 업그레이드 시 기존 파일 정리 필요

## 🔮 향후 개선 방향

### 1. **버전 관리**
- 분석 결과의 버전 히스토리 관리
- 롤백 기능 구현

### 2. **증분 업데이트**
- 특정 시각화 타입만 업데이트하는 기능
- 부분 분석 결과 병합

### 3. **분석 품질 메트릭**
- 분석 결과의 품질 점수 계산
- 품질이 낮은 분석 자동 재생성

---

**구현 완료**: ✅ 자동 분석 결과 로딩 시스템이 성공적으로 구현되었습니다.
**사용자 요구사항 충족**: ✅ "이미 한게 있으면 그대로 불러오게" 요청이 완벽하게 구현되었습니다.
