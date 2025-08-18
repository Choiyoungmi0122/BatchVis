from fastapi import APIRouter, Request, Form, Body
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import os
import json
from datetime import datetime
import pytz
from tempfile import NamedTemporaryFile

router = APIRouter(prefix="/experiment", tags=["experiment"])

RESULTS_DIR = "responses"
templates = Jinja2Templates(directory="frontend/templates")

@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/start")
async def start_experiment():
    """
    한국 시간 기준으로 날짜와 시간을 포함한 실험 번호를 생성
    형식: YYYYMMDD_HHMMSS (예: 20250720_143052)
    """
    # 한국 시간대 설정
    korea_tz = pytz.timezone('Asia/Seoul')
    korea_time = datetime.now(korea_tz)
    
    # 실험 번호 생성 (YYYYMMDD_HHMMSS 형식)
    experiment_num = korea_time.strftime('%Y%m%d_%H%M%S')
    
    return {"experiment_num": experiment_num}

@router.get("/{experiment_num}")
async def get_experiment(experiment_num: str):
    """
    특정 실험 번호의 데이터를 반환
    """
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}.json")
    if not os.path.exists(filename):
        return JSONResponse(status_code=404, content={"error": "Experiment not found"})
    
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@router.get("/status/{experiment_num}")
async def get_experiment_status(experiment_num: str):
    """
    특정 실험의 진행 상태를 반환
    """
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}.json")
    if not os.path.exists(filename):
        return JSONResponse(status_code=404, content={"error": "Experiment not found"})
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 가장 최근 history entry의 상태 확인
        if data.get("history") and len(data["history"]) > 0:
            latest_entry = data["history"][-1]
            status = latest_entry.get("status", "unknown")
            
            # 진행 상황 계산
            total_instructions = len(latest_entry.get("prompts", []))
            completed_instructions = sum(1 for p in latest_entry.get("prompts", []) if p.get("status") == "completed")
            
            return {
                "experiment_num": experiment_num,
                "status": status,
                "total_instructions": total_instructions,
                "completed_instructions": completed_instructions,
                "progress_percentage": round((completed_instructions / total_instructions * 100) if total_instructions > 0 else 0, 2),
                "latest_entry": latest_entry
            }
        else:
            return {
                "experiment_num": experiment_num,
                "status": "no_data",
                "total_instructions": 0,
                "completed_instructions": 0,
                "progress_percentage": 0
            }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/input/{filename}")
async def get_experiment_input(filename: str):
    """
    특정 실험 파일의 user_input과 instructions를 반환
    """
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # history에서 가장 첫 user_input과 prompts 반환
        if data.get("history") and len(data["history"]) > 0:
            user_input = data["history"][0].get("user_input", "")
            prompts = data["history"][0].get("prompts", [])
            return {
                "user_input": user_input,
                "prompts": prompts
            }
        else:
            return {"user_input": "", "prompts": []}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/list")
async def list_experiments():
    """
    responses 폴더의 실험 파일 목록을 읽고, (날짜, 이름, 나이)만 추출해 리스트로 반환
    """
    result = []
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if not fname.endswith('.json') or fname == 'personality.json':
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
                date = data.get('experiment_num', fname)
                # history에서 가장 첫 user_input만 사용
                user_input = ''
                if isinstance(data.get('history'), list) and data['history']:
                    user_input = data['history'][0].get('user_input', '')
                parts = [p.strip() for p in user_input.split(',')]
                name = parts[0] if len(parts) > 0 else ''
                age = parts[1] if len(parts) > 1 else ''
                symptom = parts[3] if len(parts) > 3 else ''
                result.append({
                    'filename': fname,
                    'date': date,
                    'name': name,
                    'age': age,
                    'symptom': symptom
                })
        except Exception as e:
            continue
    return result

@router.post("/save")
async def save_experiment(data: dict = Body(...)):
    """
    프론트엔드에서 실험 완료 시 전체 실험 데이터를 JSON으로 받아 저장합니다.
    파일명: experiment_{experiment_num}_{timestamp}.json
    """
    experiment_num = data.get("experiment_num", "unknown")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}_{timestamp}.json")
    
    try:
        # JSON 직렬화 가능성 확인
        json.dumps(data, ensure_ascii=False, indent=2)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] 실험 데이터 저장 완료: {filename}")
        return {"message": "실험 데이터가 저장되었습니다.", "filename": filename}
        
    except json.JSONEncodeError as e:
        error_msg = f"JSON 직렬화 오류: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return JSONResponse(status_code=500, content={"error": error_msg})
    except PermissionError as e:
        error_msg = f"파일 권한 오류: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return JSONResponse(status_code=500, content={"error": error_msg})
    except OSError as e:
        error_msg = f"파일 시스템 오류: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return JSONResponse(status_code=500, content={"error": error_msg})
    except Exception as e:
        error_msg = f"알 수 없는 저장 오류: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return JSONResponse(status_code=500, content={"error": error_msg})
