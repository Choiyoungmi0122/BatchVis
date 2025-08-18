import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import base64
import io
import time
import asyncio
import concurrent.futures
from matplotlib import pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import re
from datetime import datetime
# 저장 관련 import 제거됨

# Batch 처리 설정 가져오기
try:
    from batch_config import (
        OPENAI_CONFIG, BATCH_CONFIG, MONITORING_CONFIG, 
        FALLBACK_CONFIG, MEMORY_CONFIG, ERROR_HANDLING_CONFIG,
        get_optimal_batch_size, should_use_batch_processing
    )
    print("✅ Batch 설정 파일 로드 성공")
except ImportError:
    print("⚠️ batch_config.py를 찾을 수 없습니다. 기본 설정을 사용합니다.")
    # 기본 설정 (fallback)
    OPENAI_CONFIG = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 1000,
        "timeout": 30,
    }
    BATCH_CONFIG = {
        "default_batch_size": 10,
        "max_batch_size": 20,
        "min_batch_size": 5,
        "api_call_delay": 0.1,
        "max_retries": 3,
        "retry_delay": 1.0,
    }
    MONITORING_CONFIG = {"enable_logging": True}
    FALLBACK_CONFIG = {"enable_rule_based_fallback": True}
    MEMORY_CONFIG = {"max_concurrent_batches": 3}
    ERROR_HANDLING_CONFIG = {"continue_on_partial_failure": True}
    
    def get_optimal_batch_size(response_count: int) -> int:
        return min(10, response_count)
    
    def should_use_batch_processing(response_count: int) -> bool:
        return response_count >= 5

# 환경 변수 로드
load_dotenv()

# 문장 유사도 모델 초기화
try:
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Sentence Transformers 모델 로드 성공")
except Exception as e:
    print(f"❌ Sentence Transformers 모델 로드 실패: {str(e)}")
    sentence_model = None
    
# 저장 관련 함수들 제거됨

# 전역 저장소 인스턴스
# 저장 기능은 제거됨

class LLMTagger:
    """LLM을 사용한 의미 태깅 시스템"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = OPENAI_CONFIG.get("model", "gpt-3.5-turbo")
        self.batch_size = BATCH_CONFIG.get("default_batch_size", 10)
        self.temperature = OPENAI_CONFIG.get("temperature", 0.1)
        self.max_tokens = OPENAI_CONFIG.get("max_tokens", 1000)
        self.timeout = OPENAI_CONFIG.get("timeout", 30)
    
    def tag_response(self, response_text: str) -> Dict[str, str]:
        """LLM을 사용하여 응답 텍스트를 의미 태깅"""
        try:
            prompt = f"""
다음 문장을 감정 방향, 감정 강도, 행동 성향, 관계 지향성, 지원 요청 여부로 분류해주세요.

문장: "{response_text}"

결과 형식:
- 감정 방향: 긍정/부정/중립
- 감정 강도: 약함/보통/강함
- 행동 성향: 수동적/능동적/회피적
- 관계 지향성: 자기중심/타인지향/균형
- 지원 요청 여부: 암시적 요청/명시적 요청/없음

JSON 형식으로만 응답해주세요:
{{
    "감정 방향": "중립",
    "감정 강도": "보통",
    "행동 성향": "수동적",
    "관계 지향성": "자기중심",
    "지원 요청 여부": "없음"
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 한국어 응답을 분석하는 전문가입니다. 정확하고 일관된 태깅을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            # JSON 응답 파싱
            content = response.choices[0].message.content
            print(f"🔍 LLM 원본 응답: {content}")
            
            # JSON 블록 제거
            if content.startswith('```json'):
                content = content[7:-3]  # ```json 제거
            elif content.startswith('```'):
                content = content[3:-3]  # ``` 제거
            
            print(f"🔍 JSON 파싱 시도: {content}")
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON 파싱 실패: {str(json_error)}")
                print(f"🔍 문제가 있는 JSON: {content}")
                
                # JSON 수정 시도
                try:
                    # 따옴표 문제 수정
                    content = content.replace('"', '"').replace('"', '"')
                    # 줄바꿈 제거
                    content = content.replace('\n', '').replace('\r', '')
                    # 공백 정리
                    content = content.strip()
                    
                    print(f"🔍 수정된 JSON: {content}")
                    result = json.loads(content)
                    return result
                except:
                    print(f"❌ JSON 수정 실패, 규칙 기반 태깅으로 대체")
                    return self._rule_based_tagging(response_text)
            
        except Exception as e:
            print(f"❌ LLM 태깅 실패: {str(e)}")
            # LLM 실패 시 규칙 기반 태깅으로 대체
            return self._rule_based_tagging(response_text)
    
    def tag_responses_batch(self, responses: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """여러 응답을 병렬 Batch 처리하여 고속 의미 태깅"""
        try:
            if not responses:
                return []
            
            print(f"🚀 고속 병렬 Batch 태깅 시작: {len(responses)}개 응답")
            
            # 최적 배치 크기 계산 (더 큰 배치로 처리)
            optimal_batch_size = get_optimal_batch_size(len(responses))
            batches = [responses[i:i + optimal_batch_size] for i in range(0, len(responses), optimal_batch_size)]
            
            print(f"📦 배치 구성: {len(batches)}개 배치, 배치당 {optimal_batch_size}개 응답")
            
            # 병렬 처리를 위한 ThreadPoolExecutor 사용
            max_workers = min(MEMORY_CONFIG.get("max_concurrent_batches", 5), len(batches))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 모든 배치를 동시에 제출
                future_to_batch = {
                    executor.submit(self._process_batch, batch, batch_idx): batch_idx 
                    for batch_idx, batch in enumerate(batches)
                }
                
                # 결과 수집
                all_results = []
                completed_batches = 0
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        completed_batches += 1
                        print(f"✅ Batch {batch_idx + 1}/{len(batches)} 완료 ({completed_batches}/{len(batches)})")
                    except Exception as e:
                        print(f"❌ Batch {batch_idx + 1} 처리 실패: {str(e)}")
                        # 실패한 배치는 규칙 기반 태깅으로 대체
                        batch = batches[batch_idx]
                        fallback_results = [self._rule_based_tagging(resp['text']) for resp in batch]
                        all_results.extend(fallback_results)
                        completed_batches += 1
            
            print(f"🎉 고속 병렬 Batch 태깅 완료: {len(all_results)}개 응답 처리됨")
            
            # 메모리 정리
            if MEMORY_CONFIG.get("enable_garbage_collection", True):
                import gc
                gc.collect()
                print("🧹 메모리 정리 완료")
            
            return all_results
            
        except Exception as e:
            print(f"❌ 병렬 Batch 태깅 실패: {str(e)}")
            print("🔄 순차 처리로 대체...")
            return self._sequential_fallback(responses)
    
    def _process_batch(self, batch: List[Dict[str, str]], batch_idx: int) -> List[Dict[str, str]]:
        """개별 배치 처리 (병렬 실행용)"""
        try:
            print(f"📦 Batch {batch_idx + 1} 처리 시작 ({len(batch)}개 응답)")
            
            # Batch용 프롬프트 생성
            batch_prompt = self._create_batch_prompt(batch)
            
            # OpenAI API 호출 (최적화된 재시도 로직)
            max_retries = BATCH_CONFIG.get("max_retries", 2)
            retry_delay = BATCH_CONFIG.get("retry_delay", 0.5)
            
            for retry_attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "당신은 한국어 응답을 분석하는 전문가입니다. 여러 응답을 정확하고 일관되게 태깅해주세요."},
                            {"role": "user", "content": batch_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    break  # 성공 시 루프 탈출
                except Exception as e:
                    if retry_attempt < max_retries - 1:
                        print(f"⚠️ Batch {batch_idx + 1} API 호출 실패 (시도 {retry_attempt + 1}/{max_retries}): {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        print(f"❌ Batch {batch_idx + 1} 최대 재시도 횟수 초과")
                        raise e
            
            # Batch 결과 파싱
            content = response.choices[0].message.content
            batch_results = self._parse_batch_response(content, len(batch))
            
            # 결과 검증 및 보완
            for i, result in enumerate(batch_results):
                if not result or not isinstance(result, dict):
                    print(f"⚠️ Batch {batch_idx + 1}, 응답 {i + 1} 파싱 실패, 규칙 기반 태깅으로 대체")
                    batch_results[i] = self._rule_based_tagging(batch[i]['text'])
            
            print(f"✅ Batch {batch_idx + 1} 처리 완료")
            return batch_results
            
        except Exception as e:
            print(f"❌ Batch {batch_idx + 1} 처리 중 오류: {str(e)}")
            # 오류 발생 시 규칙 기반 태깅으로 대체
            return [self._rule_based_tagging(resp['text']) for resp in batch]
    
    def _sequential_fallback(self, responses: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """순차 처리 폴백 (병렬 처리 실패 시)"""
        print("🔄 순차 처리로 대체 중...")
        results = []
        for i, resp in enumerate(responses):
            try:
                result = self.tag_response(resp['text'])
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"📝 순차 처리 진행률: {i + 1}/{len(responses)}")
            except Exception as e:
                print(f"⚠️ 응답 {i + 1} 처리 실패, 규칙 기반 태깅으로 대체: {str(e)}")
                result = self._rule_based_tagging(resp['text'])
                results.append(result)
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        return {
            "batch_size": BATCH_CONFIG.get("default_batch_size", 25),
            "max_concurrent": MEMORY_CONFIG.get("max_concurrent_batches", 5),
            "api_delay": BATCH_CONFIG.get("api_call_delay", 0.05),
            "retry_count": BATCH_CONFIG.get("max_retries", 2)
        }
    

    
    def _create_batch_prompt(self, responses: List[Dict[str, str]]) -> str:
        """Batch 처리를 위한 프롬프트 생성"""
        prompt = """다음 응답들을 감정 방향, 감정 강도, 행동 성향, 관계 지향성, 지원 요청 여부로 분류해주세요.

각 응답에 대해 JSON 형식으로만 응답해주세요.

응답들:
"""
        
        for i, resp in enumerate(responses):
            prompt += f"{i + 1}. {resp['text']}\n"
        
        prompt += """
결과 형식:
[
    {
        "감정 방향": "긍정/부정/중립",
        "감정 강도": "약함/보통/강함",
        "행동 성향": "수동적/능동적/회피적",
        "관계 지향성": "자기중심/타인지향/균형",
        "지원 요청 여부": "암시적 요청/명시적 요청/없음"
    },
    ...
]
"""
        return prompt
    
    def _parse_batch_response(self, content: str, expected_count: int) -> List[Dict[str, str]]:
        """Batch 응답 파싱"""
        try:
            # JSON 블록 제거
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            # JSON 파싱
            result = json.loads(content)
            
            if isinstance(result, list) and len(result) == expected_count:
                return result
            else:
                print(f"⚠️ Batch 응답 형식 오류: 예상 {expected_count}개, 실제 {len(result) if isinstance(result, list) else 'not list'}")
                return []
                
        except json.JSONDecodeError as e:
            print(f"❌ Batch 응답 파싱 실패: {str(e)}")
            print(f"🔍 문제가 있는 내용: {content}")
            return []
        except Exception as e:
            print(f"❌ Batch 응답 처리 실패: {str(e)}")
            return []
    
    def _rule_based_tagging(self, response_text: str) -> Dict[str, str]:
        """규칙 기반 태깅 (LLM 실패 시 대체)"""
        result = {}
        
        # 감정 방향
        positive_words = ['행복', '기쁨', '만족', '희망', '감사', '좋다', '훌륭하다']
        negative_words = ['슬픔', '화남', '우울', '불안', '짜증', '나쁘다', '힘들다']
        
        positive_count = sum(response_text.count(word) for word in positive_words)
        negative_count = sum(response_text.count(word) for word in negative_words)
        
        if positive_count > negative_count:
            result["감정 방향"] = "긍정"
        elif negative_count > positive_count:
            result["감정 방향"] = "부정"
        else:
            result["감정 방향"] = "중립"
        
        # 감정 강도
        emotion_words = positive_words + negative_words
        emotion_count = sum(response_text.count(word) for word in emotion_words)
        
        if emotion_count >= 3:
            result["감정 강도"] = "강함"
        elif emotion_count >= 1:
            result["감정 강도"] = "보통"
        else:
            result["감정 강도"] = "약함"
        
        # 행동 성향
        active_words = ['할 것이다', '하려고', '노력', '의지', '목표', '계획']
        passive_words = ['될 것 같다', '아마도', '어쩌면', '그냥', '기다린다']
        avoidant_words = ['피하다', '회피', '도망', '숨기다']
        
        active_count = sum(response_text.count(word) for word in active_words)
        passive_count = sum(response_text.count(word) for word in passive_words)
        avoidant_count = sum(response_text.count(word) for word in avoidant_words)
        
        if active_count > passive_count and active_count > avoidant_count:
            result["행동 성향"] = "능동적"
        elif avoidant_count > active_count and avoidant_count > passive_count:
            result["행동 성향"] = "회피적"
        else:
            result["행동 성향"] = "수동적"
        
        # 관계 지향성
        self_words = ['나', '내', '저', '제', '자신', '개인']
        other_words = ['우리', '함께', '사회', '공동체', '협력', '소통']
        
        self_count = sum(response_text.count(word) for word in self_words)
        other_count = sum(response_text.count(word) for word in other_words)
        
        if self_count > other_count * 2:
            result["관계 지향성"] = "자기중심"
        elif other_count > self_count * 2:
            result["관계 지향성"] = "타인지향"
        else:
            result["관계 지향성"] = "균형"
        
        # 지원 요청 여부
        explicit_request = ['도와주세요', '조언해주세요', '가르쳐주세요', '상담받고 싶어요']
        implicit_request = ['어떻게 해야 할까요', '방법을 모르겠어요', '힘들어요']
        
        explicit_count = sum(response_text.count(word) for word in explicit_request)
        implicit_count = sum(response_text.count(word) for word in implicit_request)
        
        if explicit_count > 0:
            result["지원 요청 여부"] = "명시적 요청"
        elif implicit_count > 0:
            result["지원 요청 여부"] = "암시적 요청"
        else:
            result["지원 요청 여부"] = "없음"
        
        return result

class VisualizationGenerator:
    def __init__(self):
        self.colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        self.llm_tagger = LLMTagger()  # LLM 태거 초기화
    
    def generate_radar_chart(self, data: Dict[str, List[float]], labels: List[str], title: str = "페르소나별 의미 태그 비교") -> Dict[str, Any]:
        """Plotly를 사용한 레이더 차트 생성"""
        fig = go.Figure()
        
        for i, (persona, values) in enumerate(data.items()):
            color = self.colors[i % len(self.colors)]
            # rgba 형식으로 투명도 추가 (0.4 = 40%)
            rgba_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)"
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name=f'페르소나 {persona}',
                line=dict(color=color, width=3),
                marker=dict(color=color, size=10, symbol='circle'),
                fillcolor=rgba_color,
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 5],
                    tickmode='array',
                    tickvals=[0, 1, 2, 3, 4, 5],
                    ticktext=['0', '1', '2', '3', '4', '5'],
                    tickfont=dict(size=12),
                    tickcolor='#666'
                ),
                angularaxis=dict(
                    tickfont=dict(size=14, color='#333'),
                    tickcolor='#333'
                ),
                bgcolor='#f8f9fa'
            ),
            title=dict(
                text=title,
                font=dict(size=18),
                x=0.5,
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                x=0.5,
                y=0.85,
                xanchor='center',
                yanchor='top',
                orientation='h',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#ccc',
                borderwidth=1
            ),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            margin=dict(t=100, b=50, l=50, r=50),
            height=500
        )
        
        return json.loads(fig.to_json())
    
    def generate_heatmap(self, data: Dict[str, List[float]], labels: List[str], title: str = "페르소나별 의미 분석 히트맵") -> str:
        """Seaborn을 사용한 히트맵 생성 및 base64 인코딩"""
        try:
            # matplotlib 백엔드 설정 (Windows 호환성)
            import matplotlib
            matplotlib.use('Agg')
            
            # 데이터프레임 생성
            df = pd.DataFrame(data, index=[f"페르소나 {k}" for k in data.keys()])
            df.columns = labels
            
            # 히트맵 생성
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                df, 
                annot=True, 
                cmap="RdYlGn_r", 
                linewidths=0.5,
                fmt='.2f',
                cbar_kws={'label': '값 (0-5)'}
            )
            plt.title(title, fontsize=16, pad=20)
            plt.xlabel('의미 축', fontsize=12)
            plt.ylabel('페르소나', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # 이미지를 base64로 인코딩
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            buffer.close()
            
            return image_base64
            
        except Exception as e:
            print(f"❌ 히트맵 생성 실패: {str(e)}")
            # 오류 발생 시 간단한 HTML 테이블로 대체
            return self._generate_fallback_heatmap(data, labels, title)
    
    def _generate_fallback_heatmap(self, data: Dict[str, List[float]], labels: List[str], title: str) -> str:
        """히트맵 생성 실패 시 대체 HTML 테이블 생성"""
        try:
            html_content = f"""
            <div style="text-align: center; padding: 20px;">
                <h4 style="margin-bottom: 20px; color: #495057;">{title}</h4>
                <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
                    <thead>
                        <tr style="background-color: #f8f9fa;">
                            <th style="border: 1px solid #dee2e6; padding: 12px; text-align: center;">페르소나</th>
            """
            
            for label in labels:
                html_content += f'<th style="border: 1px solid #dee2e6; padding: 12px; text-align: center;">{label}</th>'
            
            html_content += "</tr></thead><tbody>"
            
            for persona, values in data.items():
                html_content += f'<tr><td style="border: 1px solid #dee2e6; padding: 12px; font-weight: bold;">{persona}</td>'
                for value in values:
                    # 색상 강도에 따른 배경색 설정
                    intensity = min(5, max(0, value))
                    normalized = intensity / 5
                    red = int(255 * (1 - normalized))
                    green = int(255 * normalized)
                    blue = 100
                    html_content += f'<td style="border: 1px solid #dee2e6; padding: 12px; text-align: center; background-color: rgb({red},{green},{blue}); color: white; font-weight: bold;">{intensity:.1f}</td>'
                html_content += "</tr>"
            
            html_content += "</tbody></table></div>"
            
            # HTML을 base64로 인코딩 (간단한 대안)
            return base64.b64encode(html_content.encode()).decode()
            
        except Exception as e:
            print(f"❌ 대체 히트맵 생성도 실패: {str(e)}")
            return "오류: 히트맵을 생성할 수 없습니다."
    
    def generate_plotly_heatmap(self, data: Dict[str, List[float]], labels: List[str], title: str = "페르소나별 의미 분석 히트맵") -> Dict[str, Any]:
        """Plotly를 사용한 인터랙티브 히트맵 생성"""
        try:
            # 데이터프레임 생성
            df = pd.DataFrame(data, index=[f"페르소나 {k}" for k in data.keys()])
            df.columns = labels
            
            # Plotly 히트맵 생성
            fig = px.imshow(
                df, 
                color_continuous_scale="RdBu_r",  # 빨강-파랑 (역방향)
                text_auto=True,
                aspect="auto"
            )
            
            # 레이아웃 업데이트
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=18),
                    x=0.5,
                    y=0.95
                ),
                xaxis=dict(
                    title='의미 축',
                    titlefont=dict(size=14),
                    tickangle=45,
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title='페르소나',
                    titlefont=dict(size=14),
                    tickfont=dict(size=12)
                ),
                coloraxis=dict(
                    colorbar=dict(
                        title="값 (0-5)",
                        titleside="right",
                        thickness=20,
                        len=0.8
                    )
                ),
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                margin=dict(t=100, b=50, l=50, r=50),
                height=500
            )
            
            # 텍스트 표시 개선
            fig.update_traces(
                texttemplate="%{z:.2f}",
                textfont=dict(size=12, color="white"),
                hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>값: %{z:.2f}<extra></extra>'
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            print(f"❌ Plotly 히트맵 생성 실패: {str(e)}")
            # 오류 발생 시 기존 Seaborn 히트맵으로 대체
            return {"error": f"Plotly 히트맵 생성 실패: {str(e)}", "fallback": "seaborn"}
    
    def generate_sorting_chart(self, data: Dict[str, float], title: str = "의미 축 기준 정렬") -> Dict[str, Any]:
        """Plotly를 사용한 정렬 차트 생성"""
        # 값에 따라 정렬
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        personas = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]
        
        # 색상 생성 (상위 3개는 특별한 색상)
        colors = []
        for i in range(len(personas)):
            if i < 3:
                colors.append('#FFD700')  # 금색
            else:
                colors.append(self.colors[i % len(self.colors)])
        
        fig = go.Figure(data=[
            go.Bar(
                x=personas,
                y=values,
                marker_color=colors,
                text=[f'{v:.2f}' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>값: %{y:.2f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title='페르소나',
                titlefont=dict(size=14),
                tickangle=45
            ),
            yaxis=dict(
                title='값',
                titlefont=dict(size=14),
                range=[0, max(values) * 1.1]
            ),
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            height=400,
            showlegend=False
        )
        
        return json.loads(fig.to_json())
    
    def generate_side_by_side_diff(self, response_a: str, response_b: str, persona_a: str, persona_b: str, question: str) -> Dict[str, Any]:
        """Side-by-Side Diff 시각화 생성"""
        try:
            # 1. 문장 유사도 계산
            similarity_score = self._calculate_similarity(response_a, response_b)
            
            # 2. 의미 차이 분석
            analysis_a = self.analyze_response_dimensions(response_a)
            analysis_b = self.analyze_response_dimensions(response_b)
            
            # 3. 차이점 하이라이트
            diff_highlights = self._highlight_differences(response_a, response_b)
            
            # 4. 의미 축별 차이 계산
            dimension_diffs = self._calculate_dimension_differences(analysis_a, analysis_b)
            
            # 5. 시각화 데이터 구성
            diff_data = {
                'question': question,
                'persona_a': persona_a,
                'persona_b': persona_b,
                'response_a': response_a,
                'response_b': response_b,
                'similarity_score': similarity_score,
                'analysis_a': analysis_a,
                'analysis_b': analysis_b,
                'diff_highlights': diff_highlights,
                'dimension_differences': dimension_diffs,
                'summary': self._generate_diff_summary(analysis_a, analysis_b, similarity_score)
            }
            
            return diff_data
            
        except Exception as e:
            print(f"❌ Side-by-Side Diff 생성 실패: {str(e)}")
            return self._generate_fallback_diff(response_a, response_b, persona_a, persona_b, question)
    
    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """문장 유사도 계산"""
        try:
            if sentence_model:
                # Sentence Transformers 사용
                embeddings_a = sentence_model.encode([text_a])
                embeddings_b = sentence_model.encode([text_b])
                
                # 코사인 유사도 계산
                similarity = np.dot(embeddings_a[0], embeddings_b[0]) / (
                    np.linalg.norm(embeddings_a[0]) * np.linalg.norm(embeddings_b[0])
                )
                return float(similarity)
            else:
                # difflib 사용 (대체 방법)
                return SequenceMatcher(None, text_a, text_b).ratio()
                
        except Exception as e:
            print(f"❌ 유사도 계산 실패: {str(e)}")
            return 0.5  # 기본값
    
    def _highlight_differences(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """텍스트 차이점 하이라이트"""
        try:
            # 단어 단위로 분리
            words_a = re.findall(r'\w+', text_a.lower())
            words_b = re.findall(r'\w+', text_b.lower())
            
            # 공통 단어와 고유 단어 찾기
            common_words = set(words_a) & set(words_b)
            unique_a = set(words_a) - set(words_b)
            unique_b = set(words_b) - set(words_a)
            
            # 감정 관련 키워드 분류
            emotion_keywords = {
                'positive': ['행복', '기쁨', '만족', '희망', '좋다', '훌륭하다'],
                'negative': ['슬픔', '화남', '우울', '불안', '짜증', '나쁘다', '힘들다'],
                'neutral': ['보통', '일반적', '평범', '중간']
            }
            
            # 각 응답의 감정 키워드 분석
            emotion_a = self._analyze_emotion_keywords(text_a, emotion_keywords)
            emotion_b = self._analyze_emotion_keywords(text_b, emotion_keywords)
            
            return {
                'common_words': list(common_words)[:10],  # 상위 10개
                'unique_a': list(unique_a)[:10],
                'unique_b': list(unique_b)[:10],
                'emotion_a': emotion_a,
                'emotion_b': emotion_b,
                'word_count_a': len(words_a),
                'word_count_b': len(words_b)
            }
            
        except Exception as e:
            print(f"❌ 차이점 하이라이트 실패: {str(e)}")
            return {}
    
    def _analyze_emotion_keywords(self, text: str, emotion_keywords: Dict[str, List[str]]) -> Dict[str, int]:
        """감정 키워드 분석"""
        result = {}
        for emotion_type, keywords in emotion_keywords.items():
            count = sum(text.count(keyword) for keyword in keywords)
            result[emotion_type] = count
        return result
    
    def _calculate_dimension_differences(self, analysis_a: Dict[str, float], analysis_b: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """의미 축별 차이 계산"""
        dimensions = ['emotional_intensity', 'valence', 'expression_type', 'agency', 'extroversion', 'solution_offered']
        differences = {}
        
        for dim in dimensions:
            value_a = analysis_a.get(dim, 0.0)
            value_b = analysis_b.get(dim, 0.0)
            
            diff = abs(value_a - value_b)
            diff_percentage = (diff / max(value_a, value_b)) * 100 if max(value_a, value_b) > 0 else 0
            
            # 차이 정도 분류
            if diff_percentage > 50:
                diff_level = "큰 차이"
            elif diff_percentage > 25:
                diff_level = "중간 차이"
            else:
                diff_level = "작은 차이"
            
            differences[dim] = {
                'value_a': round(value_a, 2),
                'value_b': round(value_b, 2),
                'difference': round(diff, 2),
                'difference_percentage': round(diff_percentage, 1),
                'difference_level': diff_level,
                'trend': "높음" if value_a > value_b else "낮음" if value_a < value_b else "동일"
            }
        
        return differences
    
    def _generate_diff_summary(self, analysis_a: Dict[str, float], analysis_b: Dict[str, float], similarity: float) -> str:
        """차이점 요약 생성"""
        try:
            # 가장 큰 차이를 보이는 차원 찾기
            max_diff_dim = None
            max_diff = 0
            
            for dim in ['emotional_intensity', 'valence', 'expression_type', 'agency', 'extroversion', 'solution_offered']:
                diff = abs(analysis_a.get(dim, 0) - analysis_b.get(dim, 0))
                if diff > max_diff:
                    max_diff = diff
                    max_diff_dim = dim
            
            # 유사도에 따른 전체적인 평가
            if similarity > 0.8:
                overall_assessment = "매우 유사한 응답"
            elif similarity > 0.6:
                overall_assessment = "비슷한 응답"
            elif similarity > 0.4:
                overall_assessment = "중간 정도의 차이"
            else:
                overall_assessment = "매우 다른 응답"
            
            # 요약 문장 생성
            if max_diff_dim:
                dim_name = self.get_dimension_display_name(max_diff_dim)
                summary = f"{overall_assessment}입니다. 가장 큰 차이는 '{dim_name}'에서 나타납니다."
            else:
                summary = f"{overall_assessment}입니다."
            
            return summary
            
        except Exception as e:
            print(f"❌ 요약 생성 실패: {str(e)}")
            return "응답 비교 분석이 완료되었습니다."
    
    def _generate_fallback_diff(self, response_a: str, response_b: str, persona_a: str, persona_b: str, question: str) -> Dict[str, Any]:
        """Fallback diff 데이터 생성"""
        return {
            'question': question,
            'persona_a': persona_a,
            'persona_b': persona_b,
            'response_a': response_a,
            'response_b': response_b,
            'similarity_score': 0.5,
            'analysis_a': {},
            'analysis_b': {},
            'diff_highlights': {},
            'dimension_differences': {},
            'summary': '기본 비교 분석이 완료되었습니다.'
        }
    
    def analyze_response_dimensions(self, response_text: str) -> Dict[str, float]:
        """LLM 태깅과 규칙 기반 분석을 결합하여 응답 텍스트를 분석"""
        if not response_text:
            return {}
        
        # LLM 태깅 시도
        try:
            llm_tags = self.llm_tagger.tag_response(response_text)
            print(f"🔍 LLM 태깅 결과: {llm_tags}")
            
            # LLM 태깅 결과를 수치로 변환
            analysis = {}
            
            # 감정 방향: 긍정(5), 중립(3), 부정(1)
            valence_map = {"긍정": 5, "중립": 3, "부정": 1}
            analysis['valence'] = valence_map.get(llm_tags.get("감정 방향", "중립"), 3)
            
            # 감정 강도: 강함(5), 보통(3), 약함(1)
            intensity_map = {"강함": 5, "보통": 3, "약함": 1}
            analysis['emotional_intensity'] = intensity_map.get(llm_tags.get("감정 강도", "보통"), 3)
            
            # 행동 성향: 능동적(5), 수동적(3), 회피적(1)
            agency_map = {"능동적": 5, "수동적": 3, "회피적": 1}
            analysis['agency'] = agency_map.get(llm_tags.get("행동 성향", "수동적"), 3)
            
            # 관계 지향성: 타인지향(5), 균형(3), 자기중심(1)
            extroversion_map = {"타인지향": 5, "균형": 3, "자기중심": 1}
            analysis['extroversion'] = extroversion_map.get(llm_tags.get("관계 지향성", "균형"), 3)
            
            # 지원 요청 여부: 명시적(5), 암시적(3), 없음(1)
            solution_map = {"명시적 요청": 5, "암시적 요청": 3, "없음": 1}
            analysis['solution_offered'] = solution_map.get(llm_tags.get("지원 요청 여부", "없음"), 1)
            
            # 표현 스타일 (기존 규칙 기반)
            comfort_words = ['괜찮아', '힘내', '잘될 거야', '걱정마', '위로', '안심', '희망', '기대']
            info_words = ['정보', '사실', '데이터', '통계', '연구', '분석', '결과', '증거']
            command_words = ['해야', '하지마', '필요해', '중요해', '당연해', '무조건']
            
            comfort_score = sum(response_text.count(word) for word in comfort_words) * 0.6
            info_score = sum(response_text.count(word) for word in info_words) * 0.7
            command_score = sum(response_text.count(word) for word in command_words) * 0.5
            
            style_scores = [comfort_score, info_score, command_score]
            max_style = max(style_scores)
            if max_style > 0:
                if max_style == comfort_score:
                    analysis['expression_type'] = 1  # 위로형
                elif max_style == info_score:
                    analysis['expression_type'] = 3  # 정보형
                else:
                    analysis['expression_type'] = 5  # 명령형
            else:
                analysis['expression_type'] = 3  # 중립
            
            # 응답 길이 및 복잡도 (보조 지표)
            analysis['response_length'] = min(5, max(1, len(response_text) / 100))
            
            sentences = [s.strip() for s in response_text.split('.') if s.strip()]
            if sentences:
                avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
                analysis['complexity'] = min(5, max(1, avg_sentence_length / 30))
            else:
                analysis['complexity'] = 3
            
            return analysis
            
        except Exception as e:
            print(f"❌ LLM 태깅 실패, 규칙 기반 분석으로 대체: {str(e)}")
            # LLM 실패 시 기존 규칙 기반 분석 사용
            return self._legacy_analyze_response_dimensions(response_text)
    
    def analyze_responses_batch(self, responses: List[str]) -> List[Dict[str, float]]:
        """여러 응답을 Batch로 처리하여 의미 분석"""
        if not responses:
            return []
        
        start_time = time.time()
        print(f"🚀 Batch 분석 시작: {len(responses)}개 응답")
        
        # 배치 처리 사용 여부 결정
        use_batch = should_use_batch_processing(len(responses))
        if use_batch:
            print(f"📊 배치 처리 활성화 (응답 수: {len(responses)})")
        else:
            print(f"📊 개별 처리 사용 (응답 수: {len(responses)})")
        
        # 최적 배치 크기 계산
        optimal_batch_size = get_optimal_batch_size(len(responses))
        batches = [responses[i:i + optimal_batch_size] for i in range(0, len(responses), optimal_batch_size)]
        all_analyses = []
        
        for batch_idx, batch in enumerate(batches):
            print(f"📦 Batch {batch_idx + 1}/{len(batches)} 처리 중... ({len(batch)}개)")
            
            # Batch 태깅 수행
            batch_responses = [{'text': resp} for resp in batch]
            batch_tags = self.llm_tagger.tag_responses_batch(batch_responses)
            
            # 각 응답에 대해 분석 수행
            for i, (response_text, tags) in enumerate(zip(batch, batch_tags)):
                try:
                    # 태그 결과를 수치로 변환
                    analysis = {}
                    
                    # 감정 방향: 긍정(5), 중립(3), 부정(1)
                    valence_map = {"긍정": 5, "중립": 3, "부정": 1}
                    analysis['valence'] = valence_map.get(tags.get("감정 방향", "중립"), 3)
                    
                    # 감정 강도: 강함(5), 보통(3), 약함(1)
                    intensity_map = {"강함": 5, "보통": 3, "약함": 1}
                    analysis['emotional_intensity'] = intensity_map.get(tags.get("감정 강도", "보통"), 3)
                    
                    # 행동 성향: 능동적(5), 수동적(3), 회피적(1)
                    agency_map = {"능동적": 5, "수동적": 3, "회피적": 1}
                    analysis['agency'] = agency_map.get(tags.get("행동 성향", "수동적"), 3)
                    
                    # 관계 지향성: 타인지향(5), 균형(3), 자기중심(1)
                    extroversion_map = {"타인지향": 5, "균형": 3, "자기중심": 1}
                    analysis['extroversion'] = extroversion_map.get(tags.get("관계 지향성", "균형"), 3)
                    
                    # 지원 요청 여부: 명시적(5), 암시적(3), 없음(1)
                    solution_map = {"명시적 요청": 5, "암시적 요청": 3, "없음": 1}
                    analysis['solution_offered'] = solution_map.get(tags.get("지원 요청 여부", "없음"), 1)
                    
                    # 표현 스타일 (기존 규칙 기반)
                    comfort_words = ['괜찮아', '힘내', '잘될 거야', '걱정마', '위로', '안심', '희망', '기대']
                    info_words = ['정보', '사실', '데이터', '통계', '연구', '분석', '결과', '증거']
                    command_words = ['해야', '하지마', '필요해', '중요해', '당연해', '무조건']
                    
                    comfort_score = sum(response_text.count(word) for word in comfort_words) * 0.6
                    info_score = sum(response_text.count(word) for word in info_words) * 0.7
                    command_score = sum(response_text.count(word) for word in command_words) * 0.5
                    
                    style_scores = [comfort_score, info_score, command_score]
                    max_style = max(style_scores)
                    if max_style > 0:
                        if max_style == comfort_score:
                            analysis['expression_type'] = 1  # 위로형
                        elif max_style == info_score:
                            analysis['expression_type'] = 3  # 정보형
                        else:
                            analysis['expression_type'] = 1  # 명령형
                    else:
                        analysis['expression_type'] = 3  # 중립
                    
                    # 응답 길이 및 복잡도 (보조 지표)
                    analysis['response_length'] = min(5, max(1, len(response_text) / 100))
                    
                    sentences = [s.strip() for s in response_text.split('.') if s.strip()]
                    if sentences:
                        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
                        analysis['complexity'] = min(5, max(1, avg_sentence_length / 30))
                    else:
                        analysis['complexity'] = 3
                    
                    all_analyses.append(analysis)
                    
                except Exception as e:
                    print(f"⚠️ Batch {batch_idx + 1}, 응답 {i + 1} 분석 실패: {str(e)}")
                    # 실패 시 규칙 기반 분석으로 대체
                    fallback_analysis = self._legacy_analyze_response_dimensions(response_text)
                    all_analyses.append(fallback_analysis)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if MONITORING_CONFIG.get("log_performance_metrics", True):
            print(f"📊 성능 메트릭:")
            print(f"   - 총 처리 시간: {processing_time:.2f}초")
            print(f"   - 응답당 평균 시간: {processing_time/len(responses):.3f}초")
            print(f"   - 배치 처리 효율성: {'활성화' if use_batch else '비활성화'}")
        
        # 메모리 정리 (설정에 따라)
        if MEMORY_CONFIG.get("enable_garbage_collection", True):
            import gc
            gc.collect()
            print("🧹 메모리 정리 완료")
        
        print(f"✅ Batch 분석 완료: {len(all_analyses)}개 응답 처리됨")
        return all_analyses
    
    def _legacy_analyze_response_dimensions(self, response_text: str) -> Dict[str, float]:
        """기존 규칙 기반 분석 (LLM 실패 시 대체)"""
        if not response_text:
            return {}
        
        analysis = {}
        
        # 1. 감정 강도 (Emotional Intensity) - 응답이 얼마나 감정적/냉정한가
        emotion_words = {
            'high': ['행복', '기쁨', '만족', '희망', '감사', '사랑', '즐거움', '웃음', 
                    '슬픔', '화남', '우울', '불안', '짜증', '절망', '걱정', '스트레스'],
            'medium': ['평온', '차분', '보통', '일반적', '평범'],
            'low': ['무관심', '중립', '객관적', '사실적', '분석적']
        }
        
        high_emotion = sum(response_text.count(word) for word in emotion_words['high']) * 0.4
        medium_emotion = sum(response_text.count(word) for word in emotion_words['medium']) * 0.2
        low_emotion = sum(response_text.count(word) for word in emotion_words['low']) * 0.3
        
        analysis['emotional_intensity'] = min(5, max(1, high_emotion + medium_emotion + low_emotion))
        
        # 2. 정서 방향 (Valence) - 응답이 긍정/중립/부정 중 어디쯤인가
        positive_words = ['행복', '기쁨', '만족', '희망', '감사', '사랑', '즐거움', '웃음', '좋다', '훌륭하다']
        negative_words = ['슬픔', '화남', '우울', '불안', '짜증', '절망', '걱정', '스트레스', '나쁘다', '힘들다']
        
        positive_score = sum(response_text.count(word) for word in positive_words) * 0.5
        negative_score = sum(response_text.count(word) for word in negative_words) * 0.5
        
        # 1(부정) ~ 3(중립) ~ 5(긍정) 범위로 정규화
        valence_score = 3 + (positive_score - negative_score) * 0.5
        analysis['valence'] = min(5, max(1, valence_score))
        
        # 3. 표현 스타일 (Expression Type) - 응답이 위로 중심인지 정보 중심인지
        comfort_words = ['괜찮아', '힘내', '잘될 거야', '걱정마', '위로', '안심', '희망', '기대']
        info_words = ['정보', '사실', '데이터', '통계', '연구', '분석', '결과', '증거']
        command_words = ['해야', '하지마', '필요해', '중요해', '당연해', '무조건']
        
        comfort_score = sum(response_text.count(word) for word in comfort_words) * 0.6
        info_score = sum(response_text.count(word) for word in info_words) * 0.7
        command_score = sum(response_text.count(word) for word in command_words) * 0.5
        
        # 가장 높은 점수를 가진 스타일을 선택
        style_scores = [comfort_score, info_score, command_score]
        max_style = max(style_scores)
        if max_style > 0:
            if max_style == comfort_score:
                analysis['expression_type'] = 1  # 위로형
            elif max_style == info_score:
                analysis['expression_type'] = 3  # 정보형
            else:
                analysis['expression_type'] = 5  # 명령형
        else:
            analysis['expression_type'] = 3  # 중립
        
        # 4. 자기 주도성 (Agency) - 응답자가 능동적/수동적/타인 의존적인가
        active_words = ['할 것이다', '하려고', '노력', '의지', '목표', '계획', '결심', '직접', '스스로']
        passive_words = ['될 것 같다', '아마도', '어쩌면', '그냥', '그대로', '기다린다']
        dependent_words = ['도움', '의존', '상담', '조언', '가르침', '지시', '명령']
        
        active_score = sum(response_text.count(word) for word in active_words) * 0.6
        passive_score = sum(response_text.count(word) for word in passive_words) * 0.4
        dependent_score = sum(response_text.count(word) for word in dependent_words) * 0.5
        
        # 1(수동) ~ 3(중간) ~ 5(능동) 범위로 정규화
        agency_score = 3 + (active_score - passive_score - dependent_score) * 0.3
        analysis['agency'] = min(5, max(1, agency_score))
        
        # 5. 외향성 (Extroversion) - 응답이 대외 지향적인가 내향적인가
        extrovert_words = ['우리', '함께', '사회', '공동체', '협력', '소통', '이해', '공감', '친구', '가족']
        introvert_words = ['나', '개인', '혼자', '자신', '내면', '사색', '고민', '생각']
        
        extrovert_score = sum(response_text.count(word) for word in extrovert_words) * 0.5
        introvert_score = sum(response_text.count(word) for word in introvert_words) * 0.5
        
        # 1(내향적) ~ 3(중간) ~ 5(외향적) 범위로 정규화
        extroversion_score = 3 + (extrovert_score - introvert_score) * 0.4
        analysis['extroversion'] = min(5, max(1, extroversion_score))
        
        # 6. 해결 전략 제시 여부 (Solution Offered) - 구체적인 제안이 있는가
        solution_words = ['해결', '방법', '대안', '시도', '실행', '적응', '극복', '제안', '권장', '계획']
        indirect_words = ['생각해보자', '고민해보자', '시간이 필요하다', '차근차근']
        
        direct_solution = sum(response_text.count(word) for word in solution_words) * 0.8
        indirect_solution = sum(response_text.count(word) for word in indirect_words) * 0.4
        
        if direct_solution > 0:
            analysis['solution_offered'] = 5  # 명확
        elif indirect_solution > 0:
            analysis['solution_offered'] = 3  # 간접적
        else:
            analysis['solution_offered'] = 5  # 없음
        
        # 응답 길이 및 복잡도 (보조 지표)
        analysis['response_length'] = min(5, max(1, len(response_text) / 100))
        
        sentences = [s.strip() for s in response_text.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            analysis['complexity'] = min(5, max(1, avg_sentence_length / 30))
        else:
            analysis['complexity'] = 3
        
        return analysis
    
    def get_dimension_display_name(self, dimension: str) -> str:
        """의미 축의 표시 이름을 반환"""
        display_names = {
            'emotional_intensity': '감정 강도',
            'valence': '정서 방향',
            'expression_type': '표현 스타일',
            'agency': '자기 주도성',
            'extroversion': '외향성',
            'solution_offered': '지원 요청',
            'response_length': '응답 길이',
            'complexity': '응답 복잡도'
        }
        
        if dimension.startswith('temperament_'):
            key = dimension.replace('temperament_', '')
            return f'기질: {key}'
        elif dimension.startswith('character_'):
            key = dimension.replace('character_', '')
            return f'성격: {key}'
        
        return display_names.get(dimension, dimension)

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# 전역 인스턴스 (import 후에 생성)

router = APIRouter(prefix="/visualization", tags=["visualization"])

# 새로운 구조화된 JSON 스키마 모델들
class PersonaTag(BaseModel):
    """페르소나별 의미 태그"""
    감정_방향: str = Field(..., description="긍정/부정/중립")
    감정_강도: str = Field(..., description="약함/보통/강함")
    행동_성향: str = Field(..., description="수동적/능동적/회피적")
    관계_지향성: str = Field(..., description="자기중심/타인지향/균형")
    지원_요청_여부: str = Field(..., description="암시적 요청/명시적 요청/없음")
    표현_스타일: str = Field(..., description="위로형/정보형/명령형")

class PersonaResponse(BaseModel):
    """페르소나별 응답 데이터"""
    persona: str = Field(..., description="페르소나 이름")
    text: str = Field(..., description="응답 텍스트")
    embedding: Optional[List[float]] = Field(None, description="문장 임베딩 벡터")
    tags: PersonaTag = Field(..., description="의미 태그")
    analysis_scores: Dict[str, float] = Field(..., description="의미 축별 수치 점수 (0-5)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")

class SimilarityScore(BaseModel):
    """페르소나 간 유사도 점수"""
    persona_a: str = Field(..., description="페르소나 A")
    persona_b: str = Field(..., description="페르소나 B")
    score: float = Field(..., description="유사도 점수 (0-1)")
    similarity_type: str = Field(default="cosine", description="유사도 계산 방식")

class AnalysisResult(BaseModel):
    """전체 분석 결과"""
    question_id: str = Field(..., description="질문 ID")
    question_text: str = Field(..., description="질문 텍스트")
    persona_responses: List[PersonaResponse] = Field(..., description="페르소나별 응답 목록")
    similarities: List[SimilarityScore] = Field(..., description="페르소나 간 유사도 목록")
    overall_statistics: Dict[str, Any] = Field(..., description="전체 통계 정보")
    model_info: Dict[str, str] = Field(..., description="사용된 모델 정보")

class VisualizationRequest(BaseModel):
    experiment_data: dict
    question_index: int
    dimension: str
    analysis_type: str = "all"  # "radar", "heatmap", "sorting", "all"
    heatmap_type: str = "plotly"  # "plotly", "seaborn"

class DiffRequest(BaseModel):
    experiment_data: dict
    question_index: int
    persona_a: str
    persona_b: str

class StructuredAnalysisRequest(BaseModel):
    """구조화된 분석 요청"""
    experiment_data: dict
    question_index: int
    question_text: str = Field(default="", description="프론트엔드에서 전달받은 질문 텍스트")
    include_embeddings: bool = Field(default=False, description="임베딩 포함 여부")
    include_similarities: bool = Field(default=True, description="유사도 계산 포함 여부")
    analysis_depth: str = Field(default="standard", description="분석 깊이: basic/standard/detailed")

@router.post("/generate")
async def generate_visualizations(request: VisualizationRequest):
    """실험 데이터를 기반으로 시각화 생성"""
    try:
        experiment_data = request.experiment_data
        question_index = request.question_index
        dimension = request.dimension
        analysis_type = request.analysis_type
        
        # 실험명 추출
        experiment_name = experiment_data.get('experiment_name', f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        print(f"🔄 새로운 분석 생성 시작: {experiment_name}, Q{question_index}")
        
        # 응답 데이터 수집
        responses_data = {}
        dimension_values = {}
        
        print(f"🔍 디버깅: experiment_data 구조 = {type(experiment_data)}")
        print(f"🔍 디버깅: experiment_data.keys() = {list(experiment_data.keys()) if isinstance(experiment_data, dict) else 'Not a dict'}")
        
        # 더 자세한 디버깅 정보 추가
        if isinstance(experiment_data, dict):
            print(f"🔍 디버깅: experiment_data 전체 구조:")
            for key, value in experiment_data.items():
                if isinstance(value, list):
                    print(f"  - {key}: 리스트 (길이: {len(value)})")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"    첫 번째 항목 키: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    print(f"  - {key}: 딕셔너리 (키: {list(value.keys())})")
                else:
                    print(f"  - {key}: {type(value)} = {value}")
        
        # 다양한 데이터 구조 시도
        prompts_data = []
        
        # 방법 1: history[0].answers 구조 (실제 데이터 구조 - 우선 처리)
        if 'history' in experiment_data and isinstance(experiment_data['history'], list) and len(experiment_data['history']) > 0:
            print(f"🔍 디버깅: history 키 발견, 길이 = {len(experiment_data['history'])}")
            print(f"🔍 디버깅: history[0] 키들 = {list(experiment_data['history'][0].keys())}")
            if 'answers' in experiment_data['history'][0]:
                prompts_data = experiment_data['history'][0]['answers']
                print(f"🔍 디버깅: history[0].answers에서 데이터 찾음, 개수 = {len(prompts_data)}")
                if len(prompts_data) > 0:
                    print(f"🔍 디버깅: 첫 번째 answers 항목 키들 = {list(prompts_data[0].keys())}")
            elif 'prompts' in experiment_data['history'][0]:
                prompts_data = experiment_data['history'][0]['prompts']
                print(f"🔍 디버깅: history[0].prompts에서 데이터 찾음, 개수 = {len(prompts_data)}")
            else:
                print(f"🔍 디버깅: history[0]에 answers나 prompts 키가 없음")
        # 방법 2: 직접 prompts 키
        elif 'prompts' in experiment_data:
            prompts_data = experiment_data['prompts']
            print(f"🔍 디버깅: 직접 prompts 키에서 데이터 찾음, 개수 = {len(prompts_data)}")
        
        # 방법 4: experiment_data가 리스트인 경우
        elif isinstance(experiment_data, list):
            prompts_data = experiment_data
            print(f"🔍 디버깅: experiment_data가 리스트임, 길이 = {len(prompts_data)}")
        
        if prompts_data:
            print(f"🔍 디버깅: prompts_data 개수 = {len(prompts_data)}")
            
            # Batch 처리를 위한 응답 데이터 수집
            batch_responses = []
            personality_mapping = {}
            
            for i, prompt in enumerate(prompts_data):
                print(f"🔍 디버깅: prompt[{i}] 구조 = {type(prompt)}")
                print(f"🔍 디버깅: prompt[{i}].keys() = {list(prompt.keys()) if isinstance(prompt, dict) else 'Not a dict'}")
                
                personality = prompt.get('personality', f'Unknown_{i}')
                print(f"🔍 디버깅: personality = {personality}")
                
                # prompt 전체 내용 출력 (디버깅용)
                print(f"🔍 디버깅: prompt[{i}] 전체 내용 = {prompt}")
                
                # 다양한 응답 데이터 구조 시도
                response_text = ""
                
                # 방법 1: answer 필드 찾기 (현재 데이터 구조에 맞음)
                if 'answer' in prompt:
                    response_text = prompt['answer']
                    print(f"🔍 디버깅: answer 필드 찾음 = {response_text[:50]}...")
                    print(f"🔍 디버깅: answer 필드 타입 = {type(response_text)}")
                    print(f"🔍 디버깅: answer 필드 길이 = {len(str(response_text))}")
                    print(f"🔍 디버깅: answer 필드 내용 = '{response_text}'")
                    print(f"🔍 디버깅: answer 필드가 비어있나? = {not response_text or not response_text.strip()}")
                
                # 방법 2: 기타 응답 필드들 (대안)
                elif 'response' in prompt:
                    response_text = prompt['response']
                    print(f"🔍 디버깅: response 필드 찾음 = {response_text[:50]}...")
                elif 'text' in prompt:
                    response_text = prompt['text']
                    print(f"🔍 디버깅: text 필드 찾음 = {response_text[:50]}...")
                
                print(f"🔍 디버깅: 최종 response_text 길이 = {len(response_text)}")
                
                if response_text and response_text.strip():
                    # Batch 처리를 위한 응답 텍스트 수집
                    batch_responses.append(response_text)
                    personality_mapping[len(batch_responses) - 1] = personality
                else:
                    print(f"🔍 디버깅: {personality} 응답 텍스트 없음")
            
            # Batch 분석 수행
            if batch_responses:
                print(f"🚀 Batch 분석 시작: {len(batch_responses)}개 응답")
                batch_analyses = viz_generator.analyze_responses_batch(batch_responses)
                
                # 결과를 personality별로 매핑
                for i, analysis in enumerate(batch_analyses):
                    if i in personality_mapping:
                        personality = personality_mapping[i]
                        responses_data[personality] = analysis
                        
                        # 선택된 차원의 값 추출
                        if dimension in analysis:
                            dimension_values[personality] = analysis[dimension]
                        else:
                            dimension_values[personality] = np.random.uniform(1, 4)  # 기본값
                        
                        print(f"🔍 디버깅: {personality} 응답 분석 완료, 차원값 = {dimension_values[personality]}")
            else:
                print(f"🔍 디버깅: 분석할 응답이 없음")
        else:
            print(f"🔍 디버깅: prompts 데이터를 찾을 수 없음")
        
        print(f"🔍 디버깅: 최종 responses_data 개수 = {len(responses_data)}")
        print(f"🔍 디버깅: 최종 dimension_values 개수 = {len(dimension_values)}")
        
        if not responses_data:
            if not prompts_data:
                print("❌ 응답 데이터를 찾을 수 없습니다: answers 배열이 비어있습니다")
                # 빈 결과 반환하여 화면에 오류 메시지가 뜨지 않도록 함
                return {
                    "success": True,
                    "data": {
                        "sorting_chart": None,
                        "sorting_list": None
                    },
                    "message": "분석할 응답 데이터가 없습니다"
                }
            else:
                print("❌ 응답 데이터를 찾을 수 없습니다: 응답 텍스트를 추출할 수 없습니다")
                # 빈 결과 반환하여 화면에 오류 메시지가 뜨지 않도록 함
                return {
                    "success": True,
                    "data": {
                        "sorting_chart": None,
                        "sorting_list": None
                    },
                    "message": "응답 텍스트를 추출할 수 없습니다"
                }
        
        result = {}
        
        # 레이더 차트 생성
        if analysis_type in ["radar", "all"]:
            try:
                print(f"🔍 레이더 차트 디버깅: responses_data 키 개수 = {len(responses_data)}")
                print(f"🔍 레이더 차트 디버깅: 첫 번째 응답 데이터 키 = {list(responses_data.keys())[:3]}")
                
                # 모든 차원에 대한 데이터 준비
                all_dimensions = ['emotional_intensity', 'valence', 'expression_type', 
                                'agency', 'extroversion', 'solution_offered', 'response_length', 'complexity']
                
                radar_data = {}
                for persona, analysis in responses_data.items():
                    values = []
                    for dim in all_dimensions:
                        values.append(analysis.get(dim, 0.0))  # 기본값 0.0
                    radar_data[persona] = values
                    print(f"🔍 레이더 차트 디버깅: {persona} = {values}")
                
                print(f"🔍 레이더 차트 디버깅: radar_data 키 개수 = {len(radar_data)}")
                print(f"🔍 레이더 차트 디버깅: 첫 번째 값 길이 = {len(list(radar_data.values())[0])}")
                
                result['radar_chart'] = viz_generator.generate_radar_chart(
                    radar_data, 
                    all_dimensions,
                    f"질문 {question_index + 1}: 페르소나별 의미 태그 비교"
                )
            except Exception as e:
                print(f"❌ 레이더 차트 생성 실패: {str(e)}")
                # 오류 발생 시 기본 레이더 차트 데이터 생성
                fallback_radar = {}
                for persona in responses_data.keys():
                    fallback_radar[persona] = [2.0, 2.5, 3.0, 2.5, 2.0, 2.5, 3.0, 2.5]
                
                result['radar_chart'] = viz_generator.generate_radar_chart(
                    fallback_radar,
                    ['감정 강도', '정서 방향', '표현 스타일', '자기 주도성', '외향성', '해결 전략', '응답 길이', '복잡도'],
                    f"질문 {question_index + 1}: 페르소나별 의미 태그 비교 (기본값)"
                )
        
        # 히트맵 생성
        if analysis_type in ["heatmap", "all"]:
            try:
                print(f"🔍 히트맵 디버깅: responses_data 키 개수 = {len(responses_data)}")
                print(f"🔍 히트맵 디버깅: 첫 번째 응답 데이터 키 = {list(responses_data.keys())[:3]}")
                
                heatmap_data = {}
                for persona, analysis in responses_data.items():
                    # 모든 차원에 대한 값이 있는지 확인하고, 없으면 0으로 채움
                    all_dimensions = ['emotional_intensity', 'valence', 'expression_type', 
                                    'agency', 'extroversion', 'solution_offered', 'response_length', 'complexity']
                    values = []
                    for dim in all_dimensions:
                        values.append(analysis.get(dim, 0.0))  # 기본값 0.0
                    heatmap_data[persona] = values
                    print(f"🔍 히트맵 디버깅: {persona} = {values}")
                
                print(f"🔍 히트맵 디버깅: heatmap_data 키 개수 = {len(heatmap_data)}")
                print(f"🔍 히트맵 디버깅: 첫 번째 값 길이 = {len(list(heatmap_data.values())[0])}")
                
                dimension_labels = all_dimensions
                
                # 히트맵 타입에 따라 다른 생성 방법 사용
                if request.heatmap_type == "plotly":
                    result['heatmap'] = viz_generator.generate_plotly_heatmap(
                        heatmap_data,
                        dimension_labels,
                        f"질문 {question_index + 1}: 페르소나별 의미 분석 (Plotly)"
                    )
                    result['heatmap_type'] = "plotly"
                else:
                    result['heatmap'] = viz_generator.generate_heatmap(
                        heatmap_data,
                        dimension_labels,
                        f"질문 {question_index + 1}: 페르소나별 의미 분석 (Seaborn)"
                    )
                    result['heatmap_type'] = "seaborn"
            except Exception as e:
                print(f"❌ 히트맵 생성 실패: {str(e)}")
                # 오류 발생 시 기본 히트맵 데이터 생성
                fallback_heatmap = {}
                for persona in responses_data.keys():
                    fallback_heatmap[persona] = [1.0, 2.0, 3.0, 2.5, 2.0, 2.5, 3.0, 2.5]
                
                # fallback 히트맵도 타입에 따라 생성
                if request.heatmap_type == "plotly":
                    result['heatmap'] = viz_generator.generate_plotly_heatmap(
                        fallback_heatmap,
                        ['감정 강도', '정서 방향', '표현 스타일', '자기 주도성', '외향성', '해결 전략', '응답 길이', '복잡도'],
                        f"질문 {question_index + 1}: 페르소나별 의미 분석 (기본값 - Plotly)"
                    )
                    result['heatmap_type'] = "plotly"
                else:
                    result['heatmap'] = viz_generator.generate_heatmap(
                        fallback_heatmap,
                        ['감정 강도', '정서 방향', '표현 스타일', '자기 주도성', '외향성', '해결 전략', '응답 길이', '복잡도'],
                        f"질문 {question_index + 1}: 페르소나별 의미 분석 (기본값 - Seaborn)"
                    )
                    result['heatmap_type'] = "seaborn"
        
        # 정렬 차트 생성
        if analysis_type in ["sorting", "all"]:
            result['sorting_chart'] = viz_generator.generate_sorting_chart(
                dimension_values,
                f"질문 {question_index + 1}: {dimension} 기준 정렬"
            )
            
            # 정렬 리스트 데이터도 추가
            sorted_personas = sorted(dimension_values.items(), key=lambda x: x[1], reverse=True)
            result['sorting_list'] = {
                'dimension': dimension,
                'dimension_name': viz_generator.get_dimension_display_name(dimension),
                'sorted_data': [
                    {
                        'rank': i + 1,
                        'personality': persona,
                        'value': round(value, 2),
                        'display_name': f'페르소나 {persona}'
                    }
                    for i, (persona, value) in enumerate(sorted_personas)
                ]
            }
        
        return {
            "success": True,
            "data": result,
            "message": "시각화가 성공적으로 생성되었습니다"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ 시각화 생성 중 오류 발생: {str(e)}")
        print(f"❌ 상세 오류 정보: {error_details}")
        raise HTTPException(status_code=500, detail=f"시각화 생성 실패: {str(e)}")

@router.post("/sample")
async def generate_sample_visualizations():
    """샘플 데이터로 시각화 생성 (테스트용)"""
    try:
        # 샘플 데이터
        sample_data = {
            'A': [4.2, 4.5, 3, 4.8, 3.2, 5.0, 3.5, 3.8],
            'B': [3.1, 2.8, 5, 2.5, 4.5, 3.0, 4.2, 3.1],
            'C': [5.0, 1.5, 1, 1.8, 2.1, 1.0, 2.8, 2.5]
        }
        
        labels = ['감정 강도', '정서 방향', '표현 스타일', '자기 주도성', '외향성', '해결 전략', '응답 길이', '복잡도']
        
        result = {
            'radar_chart': viz_generator.generate_radar_chart(
                sample_data, 
                labels,
                "샘플: 페르소나별 의미 태그 비교"
            ),
            'heatmap': viz_generator.generate_heatmap(
                sample_data,
                labels,
                "샘플: 페르소나별 의미 분석 히트맵"
            ),
            'sorting_chart': viz_generator.generate_sorting_chart(
                {k: v[0] for k, v in sample_data.items()},  # 첫 번째 차원으로 정렬
                "샘플: 감정 강도 기준 정렬"
            ),
            'sorting_list': {
                'dimension': 'agency',
                'dimension_name': '자기 주도성',
                'sorted_data': [
                    {'rank': 1, 'personality': 'A', 'value': 4.8, 'display_name': '페르소나 A'},
                    {'rank': 2, 'personality': 'B', 'value': 2.5, 'display_name': '페르소나 B'},
                    {'rank': 3, 'personality': 'C', 'value': 1.8, 'display_name': '페르소나 C'}
                ]
            }
        }
        
        return {
            "success": True,
            "data": result,
            "message": "샘플 시각화가 성공적으로 생성되었습니다"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"샘플 시각화 생성 실패: {str(e)}")

@router.post("/diff")
async def generate_side_by_side_diff(request: DiffRequest):
    """두 페르소나의 응답을 Side-by-Side로 비교 분석"""
    try:
        experiment_data = request.experiment_data
        question_index = request.question_index
        persona_a = request.persona_a
        persona_b = request.persona_b
        
        print(f"🔍 Side-by-Side Diff 디버깅: persona_a = {persona_a}, persona_b = {persona_b}")
        print(f"🔍 Side-by-Side Diff 디버깅: question_index = {question_index}")
        
        # 응답 데이터 찾기
        response_a = ""
        response_b = ""
        question_text = ""
        
        # 다양한 데이터 구조 시도
        prompts_data = []
        
        print(f"🔍 실험 데이터 구조 상세 분석:")
        print(f"🔍 최상위 키들: {list(experiment_data.keys())}")
        
        if 'history' in experiment_data and isinstance(experiment_data['history'], list) and len(experiment_data['history']) > 0:
            history_item = experiment_data['history'][0]
            print(f"🔍 history[0] 키들: {list(history_item.keys())}")
            print(f"🔍 history[0] 타입: {type(history_item)}")
            
            # 방법 1: history[0].prompts 구조 (우선 처리 - 실제 데이터 구조)
            if 'prompts' in history_item:
                prompts_data = history_item['prompts']
                print(f"🔍 디버깅: history[0].prompts에서 데이터 찾음, 개수 = {len(prompts_data)}")
                if prompts_data:
                    print(f"🔍 첫 번째 prompt 샘플: {prompts_data[0] if len(prompts_data) > 0 else 'None'}")
            
            # 방법 2: history[0].answers 구조 (이전 구조 지원)
            elif 'answers' in history_item:
                prompts_data = history_item['answers']
                print(f"🔍 디버깅: history[0].answers에서 데이터 찾음, 개수 = {len(prompts_data)}")
                if prompts_data:
                    print(f"🔍 첫 번째 answer 샘플: {prompts_data[0] if len(prompts_data) > 0 else 'None'}")
            
            # 방법 3: history[0] 자체가 데이터 배열인 경우
            elif isinstance(history_item, list):
                prompts_data = history_item
                print(f"🔍 디버깅: history[0]이 배열임, 개수 = {len(prompts_data)}")
                if prompts_data:
                    print(f"🔍 첫 번째 항목 샘플: {prompts_data[0] if len(prompts_data) > 0 else 'None'}")
            
            # 방법 4: history[0]에 다른 키들이 있는지 확인
            else:
                print(f"🔍 history[0]에서 prompts/answers를 찾을 수 없음. 다른 키들 확인:")
                for key, value in history_item.items():
                    if isinstance(value, list):
                        print(f"🔍 {key}: 리스트 (길이: {len(value)})")
                        if len(value) > 0:
                            print(f"🔍 {key}[0] 샘플: {value[0]}")
                    else:
                        print(f"🔍 {key}: {type(value)} = {value}")
        
        # 방법 5: 직접 prompts 키
        elif 'prompts' in experiment_data:
            prompts_data = experiment_data['prompts']
            print(f"🔍 디버깅: 직접 prompts 키에서 데이터 찾음, 개수 = {len(prompts_data)}")
        
        # 방법 6: 직접 answers 키
        elif 'answers' in experiment_data:
            prompts_data = experiment_data['answers']
            print(f"🔍 디버깅: 직접 answers 키에서 데이터 찾음, 개수 = {len(prompts_data)}")
        
        if prompts_data:
            # 질문 텍스트 찾기
            if 'questions' in experiment_data:
                questions = experiment_data['questions']
                if isinstance(questions, list) and len(questions) > question_index:
                    question_text = questions[question_index]
                elif isinstance(questions, dict) and 'prompts' in questions:
                    prompts = questions['prompts']
                    if isinstance(prompts, list) and len(prompts) > question_index:
                        question_text = prompts[question_index].get('text', f'질문 {question_index + 1}')
            
            # 각 페르소나의 응답 찾기
            print(f"🔍 페르소나 A ({persona_a})와 B ({persona_b})의 응답을 찾는 중...")
            
            for i, prompt in enumerate(prompts_data):
                # 다양한 필드명으로 페르소나 식별
                personality = prompt.get('personality', '') or prompt.get('persona', '') or prompt.get('name', '')
                print(f"🔍 [{i}] Prompt 분석: personality='{personality}', keys={list(prompt.keys())}")
                
                if personality == persona_a:
                    print(f"🔍 페르소나 A ({persona_a}) 발견! 응답 필드 찾는 중...")
                    # 방법 1: answer 필드 찾기
                    if 'answer' in prompt and prompt['answer']:
                        response_a = prompt['answer']
                        print(f"✅ 페르소나 A 응답 찾음 (answer): 길이={len(response_a)}")
                    # 방법 2: qa 배열에서 answer 찾기
                    elif 'qa' in prompt and isinstance(prompt['qa'], list) and len(prompt['qa']) > question_index:
                        qa_item = prompt['qa'][question_index]
                        if isinstance(qa_item, dict) and 'answer' in qa_item and qa_item['answer']:
                            response_a = qa_item['answer']
                            print(f"✅ 페르소나 A 응답 찾음 (qa[{question_index}].answer): 길이={len(response_a)}")
                    # 방법 3: response 필드 찾기
                    elif 'response' in prompt and prompt['response']:
                        response_a = prompt['response']
                        print(f"✅ 페르소나 A 응답 찾음 (response): 길이={len(response_a)}")
                    # 방법 4: text 필드 찾기
                    elif 'text' in prompt and prompt['text']:
                        response_a = prompt['text']
                        print(f"✅ 페르소나 A 응답 찾음 (text): 길이={len(response_a)}")
                    # 방법 5: content 필드 찾기
                    elif 'content' in prompt and prompt['content']:
                        response_a = prompt['content']
                        print(f"✅ 페르소나 A 응답 찾음 (content): 길이={len(response_a)}")
                    else:
                        print(f"❌ 페르소나 A 응답을 찾을 수 없음. 사용 가능한 필드: {list(prompt.keys())}")
                
                elif personality == persona_b:
                    print(f"🔍 페르소나 B ({persona_b}) 발견! 응답 필드 찾는 중...")
                    # 방법 1: answer 필드 찾기
                    if 'answer' in prompt and prompt['answer']:
                        response_b = prompt['answer']
                        print(f"✅ 페르소나 B 응답 찾음 (answer): 길이={len(response_b)}")
                    # 방법 2: qa 배열에서 answer 찾기
                    elif 'qa' in prompt and isinstance(prompt['qa'], list) and len(prompt['qa']) > question_index:
                        qa_item = prompt['qa'][question_index]
                        if isinstance(qa_item, dict) and 'answer' in qa_item and qa_item['answer']:
                            response_b = qa_item['answer']
                            print(f"✅ 페르소나 B 응답 찾음 (qa[{question_index}].answer): 길이={len(response_b)}")
                    # 방법 3: response 필드 찾기
                    elif 'response' in prompt and prompt['response']:
                        response_b = prompt['response']
                        print(f"✅ 페르소나 B 응답 찾음 (response): 길이={len(response_b)}")
                    # 방법 4: text 필드 찾기
                    elif 'text' in prompt and prompt['text']:
                        response_b = prompt['text']
                        print(f"✅ 페르소나 B 응답 찾음 (text): 길이={len(response_b)}")
                    # 방법 5: content 필드 찾기
                    elif 'content' in prompt and prompt['content']:
                        response_b = prompt['content']
                        print(f"✅ 페르소나 B 응답 찾음 (content): 길이={len(response_b)}")
                    else:
                        print(f"❌ 페르소나 B 응답을 찾을 수 없음. 사용 가능한 필드: {list(prompt.keys())}")
                
                # 둘 다 찾았으면 중단
                if response_a and response_b:
                    print(f"🎉 두 페르소나의 응답을 모두 찾았습니다!")
                    break
        
        print(f"🔍 Side-by-Side Diff 디버깅: response_a 길이 = {len(response_a)}")
        print(f"🔍 Side-by-Side Diff 디버깅: response_b 길이 = {len(response_b)}")
        
        # 응답을 찾지 못한 경우 대체 방법 시도
        if not response_a or not response_b:
            print(f"⚠️ 기본 방법으로 응답을 찾지 못함. 대체 방법 시도...")
            print(f"🔍 response_a 길이: {len(response_a)}, response_b 길이: {len(response_b)}")
            
            # 대체 방법: 모든 데이터를 순회하며 응답 찾기
            if not response_a or not response_b:
                print(f"🔍 대체 방법: 전체 데이터 구조에서 응답 찾기 시도...")
                
                # 실험 데이터의 모든 키를 확인
                all_keys = list(experiment_data.keys())
                print(f"🔍 실험 데이터 최상위 키들: {all_keys}")
                
                # history 외의 다른 키들도 확인
                for key in all_keys:
                    if key != 'history' and isinstance(experiment_data[key], list):
                        print(f"🔍 {key} 키에서 데이터 확인 중... (길이: {len(experiment_data[key])})")
                        
                        for item in experiment_data[key]:
                            if isinstance(item, dict):
                                # 페르소나 필드 확인
                                item_personality = item.get('personality', '') or item.get('personality', '') or item.get('name', '')
                                
                                if item_personality == persona_a and not response_a:
                                    # 응답 필드 찾기
                                    for resp_field in ['answer', 'response', 'text', 'content', 'message']:
                                        if resp_field in item and item[resp_field]:
                                            response_a = item[resp_field]
                                            print(f"✅ 대체 방법으로 페르소나 A 응답 찾음 ({resp_field}): 길이={len(response_a)}")
                                            break
                                
                                elif item_personality == persona_b and not response_b:
                                    # 응답 필드 찾기
                                    for resp_field in ['answer', 'response', 'text', 'content', 'message']:
                                        if resp_field in item and item[resp_field]:
                                            response_b = item[resp_field]
                                            print(f"✅ 대체 방법으로 페르소나 B 응답 찾음 ({resp_field}): 길이={len(response_b)}")
                                            break
                                
                                if response_a and response_b:
                                    break
                        
                        if response_a and response_b:
                            break
            
            # 여전히 찾지 못한 경우
            if not response_a or not response_b:
                print(f"❌ 모든 방법을 시도했지만 응답을 찾을 수 없음")
                print(f"🔍 최종 상태: response_a 길이={len(response_a)}, response_b 길이={len(response_b)}")
                raise HTTPException(status_code=400, detail="두 페르소나의 응답을 찾을 수 없습니다")
        
        if not question_text:
            question_text = f"질문 {question_index + 1}"
        
        # Side-by-Side Diff 생성
        diff_result = viz_generator.generate_side_by_side_diff(
            response_a, response_b, persona_a, persona_b, question_text
        )
        
        return {
            "success": True,
            "data": diff_result,
            "message": "Side-by-Side Diff가 성공적으로 생성되었습니다"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Side-by-Side Diff 생성 중 오류 발생: {str(e)}")
        print(f"❌ 상세 오류 정보: {error_details}")
        raise HTTPException(status_code=500, detail=f"Side-by-Side Diff 생성 실패: {str(e)}")

@router.post("/structured-analysis")
async def generate_structured_analysis(request: StructuredAnalysisRequest):
    """구조화된 JSON 형태로 의미 태깅 및 분석 결과 생성"""
    try:
        experiment_data = request.experiment_data
        question_index = request.question_index
        include_embeddings = request.include_embeddings
        include_similarities = request.include_similarities
        analysis_depth = request.analysis_depth
        
        # 실험명 추출
        experiment_name = experiment_data.get('experiment_name', f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        print(f"🔄 새로운 구조화된 분석 생성 시작: {experiment_name}, Q{question_index}")
        
        print(f"🔍 구조화된 분석 시작: question_index = {question_index}")
        print(f"🔍 분석 옵션: embeddings={include_embeddings}, similarities={include_similarities}, depth={analysis_depth}")
        
        # 응답 데이터 수집
        responses_data = {}
        question_text = ""
        
        # 프론트엔드에서 전달받은 질문 텍스트가 있으면 우선 사용
        if request.question_text:
            question_text = request.question_text
            print(f"✅ 프론트엔드에서 전달받은 질문 텍스트 사용: {question_text[:100]}...")
        
        # 다양한 데이터 구조 시도
        prompts_data = []
        
        print(f"🔍 구조화된 분석 디버깅: experiment_data 키들 = {list(experiment_data.keys())}")
        
        # 방법 1: history[0].answers 구조 (실제 데이터 구조 - 우선 처리)
        if 'history' in experiment_data and isinstance(experiment_data['history'], list) and len(experiment_data['history']) > 0:
            print(f"🔍 구조화된 분석 디버깅: history[0] 키들 = {list(experiment_data['history'][0].keys())}")
            if 'answers' in experiment_data['history'][0]:
                prompts_data = experiment_data['history'][0]['answers']
                print(f"🔍 디버깅: history[0].answers에서 데이터 찾음, 개수 = {len(prompts_data)}")
                if prompts_data and len(prompts_data) > 0:
                    print(f"🔍 구조화된 분석 디버깅: 첫 번째 answer 키들 = {list(prompts_data[0].keys())}")
        
        # 방법 2: history[0].prompts 구조 (대안 구조)
        elif 'history' in experiment_data and isinstance(experiment_data['history'], list) and len(experiment_data['history']) > 0:
            if 'prompts' in experiment_data['history'][0]:
                prompts_data = experiment_data['history'][0]['prompts']
                print(f"🔍 디버깅: history[0].prompts에서 데이터 찾음, 개수 = {len(prompts_data)}")
                if prompts_data and len(prompts_data) > 0:
                    print(f"🔍 구조화된 분석 디버깅: 첫 번째 prompt 키들 = {list(prompts_data[0].keys())}")
        
        # 방법 3: 직접 prompts 키
        elif 'prompts' in experiment_data:
            prompts_data = experiment_data['prompts']
            print(f"🔍 디버깅: 직접 prompts 키에서 데이터 찾음, 개수 = {len(prompts_data)}")
            if prompts_data and len(prompts_data) > 0:
                print(f"🔍 구조화된 분석 디버깅: 첫 번째 prompt 키들 = {list(prompts_data[0].keys())}")
        
        if not prompts_data:
            print(f"❌ 구조화된 분석 디버깅: prompts_data를 찾을 수 없음")
            print(f"❌ 구조화된 분석 디버깅: experiment_data 구조 = {experiment_data}")
            # 빈 결과 반환하여 화면에 오류 메시지가 뜨지 않도록 함
            return {
                "success": True,
                "data": {
                    "question_id": f"Q{question_index + 1}",
                    "question_text": f"질문 {question_index + 1}",
                    "persona_responses": [],
                    "similarities": [],
                    "overall_statistics": {},
                    "model_info": {
                        "llm_model": "N/A",
                        "embedding_model": "N/A",
                        "analysis_version": "1.0.0"
                    }
                },
                "message": "분석할 응답 데이터가 없습니다"
            }
        
        # 질문 텍스트 찾기 (프론트엔드에서 전달받은 텍스트가 있으면 우선 사용)
        if not question_text:  # 프론트엔드에서 전달받은 텍스트가 없을 때만 검색
            print(f"🔍 질문 텍스트 검색 시작: question_index={question_index}")
            
            # 방법 1: questions 배열에서 직접 검색
            if 'questions' in experiment_data:
                questions = experiment_data['questions']
                print(f"🔍 questions 키 발견: {type(questions)}")
                
                if isinstance(questions, list) and len(questions) > question_index:
                    question_text = questions[question_index]
                    print(f"✅ questions 배열에서 질문 텍스트 찾음: {question_text[:100]}...")
                elif isinstance(questions, dict):
                    print(f"🔍 questions 딕셔너리 키들: {list(questions.keys())}")
                    if 'prompts' in questions:
                        prompts = questions['prompts']
                        if isinstance(prompts, list) and len(prompts) > question_index:
                            question_text = prompts[question_index].get('text', '')
                            if question_text:
                                print(f"✅ questions.prompts에서 질문 텍스트 찾음: {question_text[:100]}...")
            
            # 방법 2: history[0].questions에서 검색
            if not question_text and 'history' in experiment_data and isinstance(experiment_data['history'], list) and len(experiment_data['history']) > 0:
                history_questions = experiment_data['history'][0].get('questions', [])
                if isinstance(history_questions, list) and len(history_questions) > question_index:
                    question_text = history_questions[question_index]
                    print(f"✅ history[0].questions에서 질문 텍스트 찾음: {question_text[:100]}...")
            
            # 방법 3: prompts_data에서 질문 찾기
            if not question_text and prompts_data:
                print(f"🔍 prompts_data에서 질문 검색 시도...")
                for prompt in prompts_data:
                    if 'question' in prompt and prompt['question']:
                        question_text = prompt['question']
                        print(f"✅ prompts_data에서 질문 텍스트 찾음: {question_text[:100]}...")
                        break
                    elif 'qa' in prompt and isinstance(prompt['qa'], list) and len(prompt['qa']) > question_index:
                        qa_item = prompt['qa'][question_index]
                        if isinstance(qa_item, dict) and 'question' in qa_item and qa_item['question']:
                            question_text = prompt['question']
                            print(f"✅ prompts_data.qa에서 질문 텍스트 찾음: {question_text[:100]}...")
                            break
            
            # 방법 4: 기본값 설정
            if not question_text:
                question_text = f"질문 {question_index + 1}"
                print(f"⚠️ 질문 텍스트를 찾을 수 없어 기본값 사용: {question_text}")
            else:
                print(f"✅ 검색으로 찾은 질문 텍스트: {question_text[:100]}...")
        else:
            print(f"✅ 프론트엔드에서 전달받은 질문 텍스트 사용: {question_text[:100]}...")
        
        # 각 페르소나의 응답 분석
        persona_responses = []
        for prompt in prompts_data:
            personality = prompt.get('personality', 'Unknown')
            
            # 응답 텍스트 찾기 (안전한 처리)
            response_text = ""
            try:
                if 'answer' in prompt and prompt['answer']:
                    response_text = str(prompt['answer'])
                elif 'qa' in prompt and isinstance(prompt['qa'], list) and len(prompt['qa']) > question_index:
                    qa_item = prompt['qa'][question_index]
                    if isinstance(qa_item, dict) and 'answer' in qa_item and qa_item['answer']:
                        response_text = str(qa_item['answer'])
                elif 'response' in prompt and prompt['response']:
                    response_text = str(prompt['response'])
                elif 'text' in prompt and prompt['text']:
                    response_text = str(prompt['text'])
                
                print(f"🔍 응답 텍스트 찾기: personality={personality}, 키={list(prompt.keys())}, response_text 길이={len(response_text) if response_text else 0}")
            except Exception as e:
                print(f"❌ 응답 텍스트 추출 실패: {str(e)}, personality={personality}")
                continue
            
            if not response_text or not response_text.strip():
                print(f"⚠️ 빈 응답 텍스트 건너뛰기: personality={personality}")
                continue
            
            # 응답 텍스트가 문자열인지 확인
            if not isinstance(response_text, str):
                print(f"⚠️ 응답 텍스트가 문자열이 아님: {type(response_text)}, personality={personality}")
                continue
            
            print(f"🔍 응답 텍스트 처리 중: personality={personality}, 길이={len(response_text)}")
            
            # LLM 태깅 수행
            llm_tags = viz_generator.llm_tagger.tag_response(response_text)
            
            # 의미 축 점수는 제거 (간단한 구조화된 분석)
            
            # 임베딩 생성 (요청된 경우)
            embedding = None
            if include_embeddings and sentence_model:
                try:
                    embedding = sentence_model.encode([response_text])[0].tolist()
                except Exception as e:
                    print(f"❌ 임베딩 생성 실패: {str(e)}")
            
            # 메타데이터는 제거 (간단한 구조화된 분석)
            
            # PersonaTag 객체 생성
            tags = PersonaTag(
                감정_방향=llm_tags.get("감정 방향", "중립"),
                감정_강도=llm_tags.get("감정 강도", "보통"),
                행동_성향=llm_tags.get("행동 성향", "수동적"),
                관계_지향성=llm_tags.get("관계 지향성", "자기중심"),
                지원_요청_여부=llm_tags.get("지원 요청 여부", "없음"),
                표현_스타일=_determine_expression_style(response_text)
            )
            
            # 의미 축 점수 계산
            analysis_scores = viz_generator.analyze_response_dimensions(response_text)
            
            # PersonaResponse 객체 생성
            persona_response = PersonaResponse(
                persona=personality,
                text=response_text,
                embedding=embedding,
                tags=tags,
                analysis_scores=analysis_scores,
                metadata={}  # 빈 딕셔너리로 설정
            )
            
            persona_responses.append(persona_response)
            responses_data[personality] = analysis_scores
        
        if not persona_responses:
            if not prompts_data:
                print("❌ 분석 가능한 응답이 없습니다: answers 배열이 비어있습니다")
            else:
                print("❌ 분석 가능한 응답이 없습니다: 응답 텍스트를 추출할 수 없습니다")
            
            # 빈 결과 반환하여 화면에 오류 메시지가 뜨지 않도록 함
            return {
                "success": True,
                "data": {
                    "question_id": f"Q{question_index + 1}",
                    "question_text": question_text,
                    "persona_responses": [],
                    "similarities": [],
                    "overall_statistics": {},
                    "model_info": {
                        "llm_model": "N/A",
                        "embedding_model": "N/A",
                        "analysis_version": "1.0.0"
                    }
                },
                "message": "분석 가능한 응답이 없습니다"
            }
        
        # 페르소나 간 유사도 계산 (요청된 경우)
        similarities = []
        if include_similarities and len(persona_responses) > 1:
            for i in range(len(persona_responses)):
                for j in range(i + 1, len(persona_responses)):
                    persona_a = persona_responses[i].persona
                    persona_b = persona_responses[j].persona
                    text_a = persona_responses[i].text
                    text_b = persona_responses[j].text
                    
                    similarity_score = viz_generator._calculate_similarity(text_a, text_b)
                    
                    similarity = SimilarityScore(
                        persona_a=persona_a,
                        persona_b=persona_b,
                        score=similarity_score,
                        similarity_type="cosine"
                    )
                    similarities.append(similarity)
        
        # 모델 정보
        model_info = {
            "llm_model": viz_generator.llm_tagger.model,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2" if sentence_model else "N/A",
            "analysis_version": "1.0.0"
        }
        
        # AnalysisResult 객체 생성
        analysis_result = AnalysisResult(
            question_id=f"Q{question_index + 1}",
            question_text=question_text,
            persona_responses=persona_responses,
            similarities=similarities,
            overall_statistics={},  # 빈 딕셔너리로 설정
            model_info=model_info
        )
        
        return {
            "success": True,
            "data": analysis_result.dict(),
            "message": "구조화된 분석이 성공적으로 완료되었습니다"
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ 구조화된 분석 생성 중 오류 발생: {str(e)}")
        print(f"❌ 상세 오류 정보: {error_details}")
        
        # 오류 발생 시에도 빈 결과 반환하여 화면에 오류 메시지가 뜨지 않도록 함
        return {
            "success": True,
            "data": {
                "question_id": f"Q{question_index + 1}",
                "question_text": f"질문 {question_index + 1}",
                "persona_responses": [],
                "similarities": [],
                "overall_statistics": {},
                "model_info": {
                    "llm_model": "N/A",
                    "embedding_model": "N/A",
                    "analysis_version": "1.0.0"
                }
            },
            "message": "분석 중 오류가 발생했습니다"
        }

def _determine_expression_style(text: str) -> str:
    """텍스트의 표현 스타일을 결정"""
    comfort_words = ['괜찮아', '힘내', '잘될 거야', '걱정마', '위로', '안심', '희망', '기대']
    info_words = ['정보', '사실', '데이터', '통계', '연구', '분석', '결과', '증거']
    command_words = ['해야', '하지마', '필요해', '중요해', '당연해', '무조건']
    
    comfort_score = sum(text.count(word) for word in comfort_words) * 0.6
    info_score = sum(text.count(word) for word in info_words) * 0.7
    command_score = sum(text.count(word) for word in command_words) * 0.5
    
    style_scores = [comfort_score, info_score, command_score]
    max_style = max(style_scores)
    
    if max_style == comfort_score:
        return "위로형"
    elif max_style == info_score:
        return "정보형"
    else:
        return "명령형"

# 전체 통계 함수 제거됨

# 저장 기능은 제거됨

# 전역 인스턴스 생성 (import 후에 생성)
viz_generator = VisualizationGenerator()
