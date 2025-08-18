#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
성능 비교 스크립트
배치 처리 vs 개별 처리의 성능 차이를 보여줍니다.
"""

import sys
import os
import time
import random

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_sample_responses(count: int) -> list:
    """샘플 응답 데이터 생성"""
    sample_texts = [
        "저는 정말 행복합니다. 매일이 즐거워요.",
        "그냥 평범한 하루였어요. 특별한 일은 없었습니다.",
        "오늘은 좀 힘들었어요. 하지만 잘 해낼 수 있을 것 같아요.",
        "새로운 계획을 세우고 있어요. 앞으로가 기대됩니다.",
        "도움이 필요해요. 누군가 조언해주실 수 있나요?",
        "감사합니다. 정말 고마워요.",
        "걱정이 많아요. 어떻게 해야 할지 모르겠어요.",
        "자신감이 생겼어요. 이제 잘 할 수 있을 것 같아요.",
        "조금 어려워요. 하지만 포기하지 않을 거예요.",
        "희망이 보여요. 앞으로 좋은 일이 있을 것 같아요."
    ]
    
    responses = []
    for i in range(count):
        # 랜덤하게 텍스트 선택하고 약간 변형
        base_text = random.choice(sample_texts)
        # 간단한 변형 (숫자나 감정어 추가)
        variations = [
            f"{base_text} {random.randint(1, 5)}번째 시도입니다.",
            f"{base_text} {random.choice(['매우', '조금', '정말'])} 그렇습니다.",
            f"{base_text} {random.choice(['오늘', '어제', '내일'])} 느낀 점입니다."
        ]
        responses.append(random.choice(variations))
    
    return responses

def simulate_individual_processing(responses: list) -> float:
    """개별 처리 시뮬레이션"""
    print(f"🔄 개별 처리 시뮬레이션: {len(responses)}개 응답")
    
    start_time = time.time()
    
    for i, response in enumerate(responses):
        # 개별 처리 시간 시뮬레이션 (API 호출 + 분석)
        processing_time = random.uniform(2.0, 4.0)  # 2-4초
        time.sleep(processing_time)
        
        if (i + 1) % 5 == 0:
            print(f"   진행률: {i + 1}/{len(responses)} ({((i + 1)/len(responses)*100):.1f}%)")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ 개별 처리 완료: {total_time:.2f}초")
    return total_time

def simulate_batch_processing(responses: list) -> float:
    """배치 처리 시뮬레이션"""
    print(f"🚀 배치 처리 시뮬레이션: {len(responses)}개 응답")
    
    start_time = time.time()
    
    # 배치 크기 계산
    batch_size = min(10, len(responses))
    batches = [responses[i:i + batch_size] for i in range(0, len(responses), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        # 배치 처리 시간 시뮬레이션 (API 호출 + 분석)
        # 배치당 기본 시간 + 응답당 추가 시간 (하지만 개별보다 효율적)
        base_time = 3.0  # 배치 기본 시간
        per_response_time = 0.3  # 응답당 추가 시간 (개별보다 빠름)
        batch_time = base_time + (len(batch) * per_response_time)
        
        time.sleep(batch_time)
        
        print(f"   배치 {batch_idx + 1}/{len(batches)} 완료: {len(batch)}개 응답, {batch_time:.2f}초")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ 배치 처리 완료: {total_time:.2f}초")
    return total_time

def calculate_performance_improvement(individual_time: float, batch_time: float) -> dict:
    """성능 향상 계산"""
    if batch_time > 0:
        improvement_factor = individual_time / batch_time
        time_saved = individual_time - batch_time
        percentage_saved = ((individual_time - batch_time) / individual_time) * 100
    else:
        improvement_factor = 0
        time_saved = 0
        percentage_saved = 0
    
    return {
        "improvement_factor": improvement_factor,
        "time_saved": time_saved,
        "percentage_saved": percentage_saved
    }

def run_performance_test(response_counts: list):
    """성능 테스트 실행"""
    print("🚀 성능 비교 테스트 시작")
    print("=" * 60)
    
    results = []
    
    for count in response_counts:
        print(f"\n📊 응답 수: {count}개")
        print("-" * 40)
        
        # 샘플 응답 생성
        responses = generate_sample_responses(count)
        
        # 개별 처리 테스트
        individual_time = simulate_individual_processing(responses)
        
        # 배치 처리 테스트
        batch_time = simulate_batch_processing(responses)
        
        # 성능 향상 계산
        improvement = calculate_performance_improvement(individual_time, batch_time)
        
        results.append({
            "response_count": count,
            "individual_time": individual_time,
            "batch_time": batch_time,
            "improvement": improvement
        })
        
        print(f"📈 성능 향상:")
        print(f"   - 개별 처리: {individual_time:.2f}초")
        print(f"   - 배치 처리: {batch_time:.2f}초")
        print(f"   - 성능 향상: {improvement['improvement_factor']:.1f}배")
        print(f"   - 시간 절약: {improvement['time_saved']:.2f}초 ({improvement['percentage_saved']:.1f}%)")
    
    return results

def display_summary(results: list):
    """결과 요약 표시"""
    print("\n" + "=" * 60)
    print("📊 성능 테스트 결과 요약")
    print("=" * 60)
    
    print(f"{'응답 수':>8} | {'개별 처리':>10} | {'배치 처리':>10} | {'성능 향상':>10} | {'시간 절약':>12}")
    print("-" * 70)
    
    total_individual = 0
    total_batch = 0
    
    for result in results:
        count = result["response_count"]
        individual = result["individual_time"]
        batch = result["batch_time"]
        improvement = result["improvement"]["improvement_factor"]
        time_saved = result["improvement"]["time_saved"]
        
        print(f"{count:>8} | {individual:>10.2f} | {batch:>10.2f} | {improvement:>10.1f} | {time_saved:>12.2f}")
        
        total_individual += individual
        total_batch += batch
    
    print("-" * 70)
    total_improvement = total_individual / total_batch if total_batch > 0 else 0
    total_time_saved = total_individual - total_batch
    total_percentage_saved = ((total_individual - total_batch) / total_individual) * 100 if total_individual > 0 else 0
    
    print(f"{'총계':>8} | {total_individual:>10.2f} | {total_batch:>10.2f} | {total_improvement:>10.1f} | {total_time_saved:>12.2f}")
    print(f"\n🎯 전체 성능 향상: {total_improvement:.1f}배")
    print(f"⏰ 총 시간 절약: {total_time_saved:.2f}초 ({total_percentage_saved:.1f}%)")

def main():
    """메인 함수"""
    print("🚀 LLM 태깅 성능 비교 테스트")
    print("배치 처리 vs 개별 처리의 성능 차이를 시뮬레이션합니다.")
    print("=" * 60)
    
    # 테스트할 응답 수들
    response_counts = [5, 10, 15, 20, 25]
    
    print(f"📋 테스트 응답 수: {response_counts}")
    print("⚠️  실제 API 호출 없이 시뮬레이션으로 진행됩니다.")
    print("   실제 환경에서는 네트워크 지연과 API 응답 시간이 추가됩니다.")
    
    # 사용자 확인
    try:
        input("\nEnter를 눌러 테스트를 시작하세요...")
    except KeyboardInterrupt:
        print("\n❌ 테스트가 취소되었습니다.")
        return
    
    try:
        # 성능 테스트 실행
        results = run_performance_test(response_counts)
        
        # 결과 요약 표시
        display_summary(results)
        
        print("\n🎉 성능 테스트 완료!")
        
    except KeyboardInterrupt:
        print("\n❌ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
