# Batch 처리 설정 파일
# LLM 태깅 성능 최적화를 위한 설정

# OpenAI API 설정
OPENAI_CONFIG = {
    "model": "gpt-3.5-turbo",  # 사용할 모델
    "temperature": 0.1,         # 응답 일관성을 위한 낮은 temperature
    "max_tokens": 1000,         # Batch 처리용 토큰 수 증가
    "timeout": 30,              # API 호출 타임아웃 (초)
}

# Batch 처리 설정 (고속 최적화)
BATCH_CONFIG = {
    "default_batch_size": 25,           # 기본 배치 크기 (증가)
    "max_batch_size": 50,               # 최대 배치 크기 (증가)
    "min_batch_size": 10,               # 최소 배치 크기 (증가)
    "api_call_delay": 0.05,             # API 호출 간격 (감소)
    "max_retries": 2,                   # 최대 재시도 횟수 (감소)
    "retry_delay": 0.5,                 # 재시도 간격 (감소)
}

# 성능 모니터링 설정
MONITORING_CONFIG = {
    "enable_logging": True,             # 로깅 활성화
    "log_batch_progress": True,         # 배치 진행률 로깅
    "log_performance_metrics": True,    # 성능 메트릭 로깅
    "performance_threshold": 5.0,       # 성능 향상 임계값 (배)
}

# Fallback 설정
FALLBACK_CONFIG = {
    "enable_rule_based_fallback": True, # 규칙 기반 태깅 대체 활성화
    "enable_individual_fallback": True, # 개별 처리 대체 활성화
    "fallback_priority": ["llm", "rule_based", "individual"], # 대체 우선순위
}

# 메모리 최적화 설정 (고속 최적화)
MEMORY_CONFIG = {
    "max_concurrent_batches": 5,        # 동시 처리할 최대 배치 수 (증가)
    "memory_cleanup_interval": 3,       # 메모리 정리 간격 (배치 수) (감소)
    "enable_garbage_collection": True,  # 가비지 컬렉션 활성화
}

# 에러 처리 설정
ERROR_HANDLING_CONFIG = {
    "continue_on_partial_failure": True, # 부분 실패 시 계속 진행
    "log_error_details": True,           # 에러 상세 정보 로깅
    "max_consecutive_failures": 3,       # 연속 실패 최대 횟수
}

# 성능 최적화 팁
PERFORMANCE_TIPS = {
    "optimal_batch_size": "응답 수가 10개 이상일 때 배치 처리 활성화",
    "api_rate_limit": "OpenAI API Rate Limit을 고려한 적절한 간격 설정",
    "memory_management": "대용량 데이터 처리 시 메모리 사용량 모니터링",
    "fallback_strategy": "LLM 실패 시 규칙 기반 분석으로 자동 전환",
}

def get_optimal_batch_size(response_count: int) -> int:
    """응답 수에 따른 최적 배치 크기 계산"""
    if response_count <= BATCH_CONFIG["min_batch_size"]:
        return response_count
    elif response_count <= BATCH_CONFIG["default_batch_size"]:
        return BATCH_CONFIG["default_batch_size"]
    elif response_count <= BATCH_CONFIG["max_batch_size"]:
        return min(response_count, BATCH_CONFIG["max_batch_size"])
    else:
        # 대용량 데이터의 경우 배치 크기 조정
        return min(BATCH_CONFIG["max_batch_size"], response_count // 2)

def should_use_batch_processing(response_count: int) -> bool:
    """배치 처리 사용 여부 결정"""
    return response_count >= BATCH_CONFIG["min_batch_size"]

def calculate_expected_processing_time(response_count: int, use_batch: bool = True) -> float:
    """예상 처리 시간 계산 (초)"""
    if use_batch and should_use_batch_processing(response_count):
        batch_size = get_optimal_batch_size(response_count)
        num_batches = (response_count + batch_size - 1) // batch_size
        # 배치당 평균 처리 시간 (초)
        avg_batch_time = 8.0  # 10개 응답 기준
        return num_batches * avg_batch_time
    else:
        # 개별 처리 시간
        return response_count * 3.0  # 응답당 평균 3초

def get_performance_improvement(response_count: int) -> float:
    """성능 향상 배수 계산"""
    individual_time = calculate_expected_processing_time(response_count, use_batch=False)
    batch_time = calculate_expected_processing_time(response_count, use_batch=True)
    
    if batch_time > 0:
        return individual_time / batch_time
    return 1.0

# 설정 검증
def validate_config() -> bool:
    """설정 유효성 검증"""
    try:
        assert BATCH_CONFIG["min_batch_size"] <= BATCH_CONFIG["default_batch_size"] <= BATCH_CONFIG["max_batch_size"]
        assert BATCH_CONFIG["api_call_delay"] >= 0
        assert BATCH_CONFIG["max_retries"] >= 0
        assert OPENAI_CONFIG["temperature"] >= 0 and OPENAI_CONFIG["temperature"] <= 2
        assert OPENAI_CONFIG["max_tokens"] > 0
        return True
    except AssertionError:
        print("❌ 설정 유효성 검증 실패")
        return False

if __name__ == "__main__":
    # 설정 테스트
    print("🔧 Batch 처리 설정 테스트")
    print(f"✅ 설정 유효성: {validate_config()}")
    
    # 성능 테스트
    test_counts = [5, 10, 20, 50, 100]
    print("\n📊 성능 비교 테스트")
    print("응답 수 | 개별 처리 | 배치 처리 | 성능 향상")
    print("-" * 50)
    
    for count in test_counts:
        individual = calculate_expected_processing_time(count, False)
        batch = calculate_expected_processing_time(count, True)
        improvement = get_performance_improvement(count)
        
        print(f"{count:6d} | {individual:8.1f}초 | {batch:8.1f}초 | {improvement:6.1f}배")
    
    print(f"\n🚀 최적 배치 크기: {BATCH_CONFIG['default_batch_size']}")
    print(f"📈 최대 성능 향상: {max(get_performance_improvement(c) for c in test_counts):.1f}배")
