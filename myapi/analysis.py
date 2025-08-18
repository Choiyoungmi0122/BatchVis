from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import json

router = APIRouter(prefix="/analysis", tags=["analysis"])

RESULTS_DIR = "responses"

@router.post("/similarity")
async def analyze_similarity(data: dict = Body(...)):
    """
    페르소나 응답 간 의미 유사도 분석
    data = {
        "experiment_files": ["experiment_20250808_165344.json", ...],  # 분석할 실험 파일들
        "analysis_type": "all" | "similarity_matrix" | "clustering" | "dimensionality_reduction"
    }
    """
    try:
        experiment_files = data.get("experiment_files", [])
        analysis_type = data.get("analysis_type", "all")
        
        if not experiment_files:
            # 모든 실험 파일 사용
            experiment_files = [f for f in os.listdir(RESULTS_DIR) 
                             if f.startswith("experiment_") and f.endswith(".json")]
        
        # 응답 데이터 수집
        responses_data = []
        for filename in experiment_files:
            filepath = os.path.join(RESULTS_DIR, filename)
            if not os.path.exists(filepath):
                continue
                
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # history에서 answers 추출
                if data.get("history"):
                    for entry in data["history"]:
                        if entry.get("answers"):
                            for answer in entry["answers"]:
                                if answer.get("answer") and answer.get("personality"):
                                    responses_data.append({
                                        "filename": filename,
                                        "experiment_num": data.get("experiment_num", ""),
                                        "personality": answer["personality"],
                                        "answer": answer["answer"],
                                        "detail": answer.get("detail", {}),
                                        "question": entry.get("question_text", "")
                                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        
        if not responses_data:
            return JSONResponse(status_code=400, content={"error": "No response data found"})
        
        # Sentence Transformer 모델 로드
        model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        
        # 응답 텍스트를 임베딩으로 변환
        texts = [item["answer"] for item in responses_data]
        embeddings = model.encode(texts, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(embeddings_np)
        
        result = {
            "total_responses": len(responses_data),
            "similarity_matrix": similarity_matrix.tolist(),
            "responses_info": responses_data
        }
        
        # 분석 타입에 따른 추가 결과
        if analysis_type in ["clustering", "all"]:
            # K-means 클러스터링
            n_clusters = min(5, len(responses_data) // 3) if len(responses_data) > 15 else 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_np)
            
            # DBSCAN 클러스터링
            dbscan = DBSCAN(eps=0.3, min_samples=2)
            dbscan_labels = dbscan.fit_predict(embeddings_np)
            
            result["clustering"] = {
                "kmeans_labels": cluster_labels.tolist(),
                "dbscan_labels": dbscan_labels.tolist(),
                "n_clusters_kmeans": n_clusters,
                "n_clusters_dbscan": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            }
        
        if analysis_type in ["dimensionality_reduction", "all"]:
            # t-SNE 차원 축소
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(responses_data)-1))
            tsne_embeddings = tsne.fit_transform(embeddings_np)
            
            # PCA 차원 축소
            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings_np)
            
            result["dimensionality_reduction"] = {
                "tsne": tsne_embeddings.tolist(),
                "pca": pca_embeddings.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
            }
        
        return result
        
    except Exception as e:
        print(f"Similarity analysis error: {e}")
        return JSONResponse(
            status_code=500, 
            content={
                "error": f"유사도 분석 실패: {str(e)}",
                "suggestion": "데이터 형식을 확인하거나 서버를 재시작해보세요."
            }
        )

@router.post("/statistics")
async def get_similarity_statistics(data: dict = Body(...)):
    """
    유사도 분석 결과에 대한 통계 정보 반환
    """
    try:
        analysis_result = data.get("analysis_result")
        if not analysis_result:
            return JSONResponse(status_code=400, content={"error": "Analysis result required"})
        
        similarity_matrix = np.array(analysis_result["similarity_matrix"])
        responses_info = analysis_result["responses_info"]
        
        # 상삼각 행렬만 사용 (자기 자신과의 유사도 제외)
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        # 기본 통계
        stats = {
            "total_responses": len(responses_info),
            "similarity_stats": {
                "mean": float(np.mean(upper_tri)),
                "median": float(np.median(upper_tri)),
                "std": float(np.std(upper_tri)),
                "min": float(np.min(upper_tri)),
                "max": float(np.max(upper_tri)),
                "q25": float(np.percentile(upper_tri, 25)),
                "q75": float(np.percentile(upper_tri, 75))
            }
        }
        
        # 가장 유사한 응답 쌍 찾기
        max_similarity_idx = np.unravel_index(np.argmax(upper_tri), upper_tri.shape)
        max_similarity = upper_tri[max_similarity_idx]
        
        # 가장 유사한 응답 쌍의 인덱스 계산
        n = similarity_matrix.shape[0]
        i, j = np.triu_indices_from(similarity_matrix, k=1)
        max_idx_i = i[max_similarity_idx[0]]
        max_idx_j = j[max_similarity_idx[0]]
        
        stats["most_similar_pair"] = {
            "similarity": float(max_similarity),
            "response1": {
                "index": int(max_idx_i),
                "personality": responses_info[max_idx_i]["personality"],
                "answer_preview": responses_info[max_idx_i]["answer"][:100] + "..."
            },
            "response2": {
                "index": int(max_idx_j),
                "personality": responses_info[max_idx_j]["personality"],
                "answer_preview": responses_info[max_idx_j]["answer"][:100] + "..."
            }
        }
        
        # 가장 덜 유사한 응답 쌍 찾기
        min_similarity_idx = np.unravel_index(np.argmin(upper_tri), upper_tri.shape)
        min_similarity = upper_tri[min_similarity_idx]
        
        min_idx_i = i[min_similarity_idx[0]]
        min_idx_j = j[min_similarity_idx[0]]
        
        stats["least_similar_pair"] = {
            "similarity": float(min_similarity),
            "response1": {
                "index": int(min_idx_i),
                "personality": responses_info[min_idx_i]["personality"],
                "answer_preview": responses_info[min_idx_i]["answer"][:100] + "..."
            },
            "response2": {
                "index": int(min_idx_j),
                "personality": responses_info[min_idx_j]["personality"],
                "answer_preview": responses_info[min_idx_j]["answer"][:100] + "..."
            }
        }
        
        # 클러스터링 통계
        if "clustering" in analysis_result:
            kmeans_labels = analysis_result["clustering"]["kmeans_labels"]
            unique_labels, counts = np.unique(kmeans_labels, return_counts=True)
            
            stats["clustering_stats"] = {
                "kmeans": {
                    "n_clusters": int(len(unique_labels)),
                    "cluster_sizes": [int(count) for count in counts],
                    "cluster_distribution": dict(zip([int(label) for label in unique_labels], 
                                                   [int(count) for count in counts]))
                }
            }
        
        return stats
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/similar_responses")
async def find_similar_responses(data: dict = Body(...)):
    """
    특정 응답과 가장 유사한 응답들을 찾기
    data = {
        "analysis_result": {...},  # analyze_similarity 결과
        "target_response_index": 0,  # 대상 응답 인덱스
        "top_k": 5  # 상위 k개 유사 응답
    }
    """
    try:
        analysis_result = data.get("analysis_result")
        target_index = data.get("target_response_index", 0)
        top_k = data.get("top_k", 5)
        
        if not analysis_result:
            return JSONResponse(status_code=400, content={"error": "Analysis result required"})
        
        similarity_matrix = np.array(analysis_result["similarity_matrix"])
        responses_info = analysis_result["responses_info"]
        
        if target_index >= len(similarity_matrix):
            return JSONResponse(status_code=400, content={"error": "Invalid target index"})
        
        # 대상 응답과의 유사도 계산
        target_similarities = similarity_matrix[target_index]
        
        # 자기 자신 제외하고 상위 k개 유사 응답 찾기
        similar_indices = np.argsort(target_similarities)[::-1][1:top_k+1]
        
        similar_responses = []
        for idx in similar_indices:
            similar_responses.append({
                "index": int(idx),
                "similarity": float(target_similarities[idx]),
                "personality": responses_info[idx]["personality"],
                "answer": responses_info[idx]["answer"],
                "question": responses_info[idx]["question"]
            })
        
        target_response = {
            "index": int(target_index),
            "personality": responses_info[target_index]["personality"],
            "answer": responses_info[target_index]["answer"],
            "question": responses_info[target_index]["question"]
        }
        
        return {
            "target_response": target_response,
            "similar_responses": similar_responses,
            "top_k": top_k
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
