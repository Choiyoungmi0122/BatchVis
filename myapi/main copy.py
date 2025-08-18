from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
from datetime import datetime
import openai
from dotenv import load_dotenv
import glob
import pytz
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed

# visualization 라우터 import
from visualization import router as visualization_router

app = FastAPI()

# visualization 라우터 포함
app.include_router(visualization_router)

origins = [
    # "http://127.0.0.1:5173",    # 또는 
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/responses", StaticFiles(directory="responses"), name="responses")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 디버깅: API 키 로드 확인
print(f"OpenAI API Key loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
if OPENAI_API_KEY:
    print(f"API Key starts with: {OPENAI_API_KEY[:10]}...")
else:
    print("WARNING: OpenAI API Key not found in .env file!")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 현재 파일이 있는 디렉토리 기준으로 responses 폴더 설정
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "responses")
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal Server Error: {str(exc)}"}
    )

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/visualization")
async def visualization_page(request: Request):
    return templates.TemplateResponse("visualization.html", {"request": request})

@app.get("/analysis")
async def analysis_page(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.post("/generate")
async def generate(trait: str = Form(...), experiment_num: str = Form(...)):
    prompt = f"Trait 조합: {trait}\n실험 번호: {experiment_num}\nGPT에게 질문하세요."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            n=3,
            max_tokens=300,
            temperature=0.7,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    result = {
        "experiment_num": experiment_num,
        "trait": trait,
        "prompt": prompt,
        "responses": [choice.message["content"] for choice in response.choices],
        "timestamp": datetime.now().isoformat(),
    }

    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return JSONResponse(content={"message": "성공적으로 저장했습니다.", "filename": filename, "responses": result["responses"]})

@app.post("/generate_virtual_patient")
async def generate_virtual_patient(data: dict = Body(...)):
    """
    data = {
        "prompt": instruction 프롬프트,
        "model": (optional, default: gpt-4o)
    }
    """
    prompt = data.get("prompt")
    model = data.get("model", "gpt-4o")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "너는 환자 인물 생성 전문가야."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        answer = response.choices[0].message.content
        return {"result": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask_patient")
async def ask_patient(data: dict = Body(...)):
    """
    data = {
        "instruction": 챗봇 instruction (가상환자 역할),
        "question": 질문,
        "model": (optional, default: gpt-4o)
    }
    """
    context = data.get("context")
    instruction = data.get("instruction")
    question = data.get("question")
    model = data.get("model", "gpt-4o")
    prompt = f"{context}\nQ: {question}\nA:"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 가상환자 역할을 수행하는 AI입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        answer = response.choices[0].message.content
        return {"result": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/save_experiment")
async def save_experiment(data: dict = Body(...)):
    """
    프론트엔드에서 실험 완료 시 전체 실험 데이터를 JSON으로 받아 저장합니다.
    파일명: experiment_{experiment_num}_{timestamp}.json
    """
    experiment_num = data.get("experiment_num", "unknown")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"message": "실험 데이터가 저장되었습니다.", "filename": filename}

@app.post("/start_experiment")
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

@app.get("/get_experiment/{experiment_num}")
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

@app.get("/get_experiment_input/{filename}")
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

@app.post("/generate_instructions")
async def generate_instructions(data: dict = Body(...)):
    """
    사용자 입력을 받아 instruction들을 단계별로 생성
    data = {
        "user_input": "홍길동, 24세, 남성, 우울증"
    }
    """
    user_input = data.get("user_input")
    
    # 1. personality.json 읽기
    personality_file = os.path.join("responses", "personality.json")
    with open(personality_file, "r", encoding="utf-8") as f:
        personality_data = json.load(f)
    
    # 2. Temperament+Character 조합만 virtual_patient_prompt 생성
    instructions = []
    temperament = [t for t in personality_data if t["type"] == "temperament"]
    character = [t for t in personality_data if t["type"] == "character"]
    for t1 in temperament:
        for t2 in character:
            d1 = t1["detail"]
            d2 = t2["detail"]
            input_parts = user_input.split(',')
            name = input_parts[0].strip() if len(input_parts) > 0 else ''
            age = input_parts[1].replace('년생','').strip() if len(input_parts) > 1 else ''
            gender = input_parts[2].strip() if len(input_parts) > 2 else ''
            symptom = input_parts[3].strip() if len(input_parts) > 3 else ''
            virtual_prompt = f"""당신은 다음 조건을 가진 가상환자입니다. 이 역할을 완전히 수행해주세요.

환자 정보:
- 이름: {name}
- 나이: {age}
- 성별: {gender}
- 주소증(주 증상): {symptom}
- TCI 성향:
- 기질(Temperament): 
  - 자극추구: {d1.get('자극추구','-')}
  - 위험회피: {d1.get('위험회피','-')}
  - 사회적민감성: {d1.get('사회적민감성','-')}
  - 인내력: {d1.get('인내력','-')}
- 성격(Character):
  - 자율성: {d2.get('자율성','-')}
  - 연대감: {d2.get('연대감','-')}
  - 자기초월: {d2.get('자기초월','-')}

[대화규칙]
1. 아래 성향 수치를 바탕으로 말투, 감정 표현, 사고 방식, 비언어적 표현(예: 한숨, 말끝 흐림, 머뭇거림 등)이 자연스럽게 드러나야 합니다.
2. 모든 응답은 1인칭 시점에서 일관되게 작성되며, 말의 길이는 상황에 맞게 간결하게 유지합니다.
3. 답변은 실제 인간처럼 감정을 느끼고 경험하는 듯한 방식으로 작성하며, TCI 성향이 응답 전반에 녹아 있어야 합니다.
4. 말끝을 흐리거나 망설이는 표현은 **가끔만** 사용하세요.  
5. 각 발화의 시작은 자연스럽고 다양하게 하세요. 
   - 매번 같은 패턴("음...", "사실...", "글쎄요..." 등)으로 시작하지 마세요.
   - 시작 표현은 상황에 따라 달라질 수 있습니다.
6. 의학 용어나 공식적 표현보다는 환자가 실제로 일상에서 사용할 법한 표현을 선택하세요.


[예시 시작 표현 — 참고용]
- "요즘 들어서 기운이 없어요."
- "딱히 큰 병은 없는데, 좀 마음이 무겁네요."
- "가끔은 아무것도 하고 싶지 않을 때가 있어요."
- "최근에 예전보다 더 피곤해지는 것 같아요."
- "사람들이랑 만나기도 좀 꺼려져요."
- "별다른 문제는 없지만, 우울한 기분이 자주 들어요."

이제 당신은 위 환자입니다. 질문에 응답하세요."""
            instructions.append({
                "type": "personality+character",
                "prompt": virtual_prompt,
                "detail": {"temperament": d1, "character": d2},
                "personality": f"{t1.get('personality','')}, {t2.get('personality','')}"
            })
    total_count = len(temperament) * len(character)
    return {
        "message": "Instruction 생성 완료",
        "instructions": instructions,
        "total_count": total_count
    }

@app.post("/process_qa")
async def process_qa(data: dict = Body(...)):
    """
    생성된 instruction들에 대해 질문-답변 처리 (병렬화)
    data = {
        "experiment_num": "20250720_123456",
        "instructions": [...],
        "user_input": "홍길동, 20살, 남성, 우울증"
    }
    """
    experiment_num = data.get("experiment_num")
    instructions = data.get("instructions")
    user_input = data.get("user_input")
    
    # questions.json 읽기
    questions_file = os.path.join("frontend", "static", "questions.json")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    prompts = []

    def get_answer(instruction, q):
        qa_prompt = f"{instruction['prompt']}\n\n질문: {q['text']}"
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 가상환자 역할을 수행하는 AI입니다."},
                    {"role": "user", "content": qa_prompt}
                ],
                max_tokens=500
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"
        return {
            "question": q["text"],
            "answer": answer
        }

    # 병렬 처리: instruction별로, 각 질문에 대해 병렬 호출
    for instruction in instructions:
        prompt_data = {
            "type": instruction["type"],
            "user_input": user_input,
            "virtual_patient_prompt": instruction["prompt"],
            "qa": [],
            "detail": instruction.get("detail", {}),
            "personality": instruction.get("personality", "")
        }
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 질문별로 병렬 실행
            futures = [executor.submit(get_answer, instruction, q) for q in questions]
            for future in as_completed(futures):
                prompt_data["qa"].append(future.result())
        # 질문 순서 보장
        prompt_data["qa"].sort(key=lambda x: [q["text"] for q in questions].index(x["question"]))
        prompts.append(prompt_data)

    # experiment_번호.json에 history로 저장 (이하 기존과 동일)
    filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_num}.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            history = existing_data.get("history", [])
    else:
        history = []
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "prompts": prompts
    }
    history.append(new_entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_num": experiment_num,
            "history": history
        }, f, ensure_ascii=False, indent=2)
    return {
        "message": "질문-답변 처리 완료",
        "experiment_num": experiment_num,
        "prompts_count": len(prompts),
        "questions_count": len(questions),
        "prompts": prompts
    }

@app.post("/process_qa_one_question")
async def process_qa_one_question(data: dict = Body(...)):
    """
    한 질문에 대해 전체 조합을 batch로 처리
    data = {
        "experiment_num": "20250720_123456",
        "instructions": [...],
        "user_input": "홍길동, 20살, 남성, 우울증",
        "question_text": "최근에 기분이 어떠셨나요?"
    }
    """
    experiment_num = data.get("experiment_num")
    instructions = data.get("instructions")
    user_input = data.get("user_input")
    question_text = data.get("question_text")

    # 각 조합별로 하나씩 OpenAI API 호출 (messages는 반드시 1개 대화만!)
    answers = []
    for inst in instructions:
        messages = [
            {"role": "system", "content": "당신은 가상환자 역할을 수행하는 AI입니다."},
            {"role": "user", "content": f"{inst['prompt']}\n\n질문: {question_text}"}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {str(e)}"
        answers.append(answer)

    # 각 조합별 답변과 detail 정보 반환
    result = []
    for idx, inst in enumerate(instructions):
        result.append({
            "type": inst.get("type", ""),
            "detail": inst.get("detail", {}),
            "personality": inst.get("personality", ""),
            "answer": answers[idx] if idx < len(answers) else "(no answer)"
        })

    return {
        "message": "질문별 batch 답변 완료",
        "answers": result,
        "question_text": question_text,
        "combination_count": len(instructions)
    }

@app.post("/process_qa_batch")
async def process_qa_batch(data: dict = Body(...)):
    """
    실험/분석용 OpenAI Batch API 대량 처리 엔드포인트
    data = {
        "experiment_num": "20250720_123456",
        "user_input": "홍길동, 24세, 남성, 우울증"
    }
    """
    experiment_num = data.get("experiment_num")
    user_input = data.get("user_input")

    # 1. 조합 데이터 로드
    personality_file = os.path.join("responses", "personality.json")
    with open(personality_file, "r", encoding="utf-8") as f:
        personality_data = json.load(f)

    # 2. 질문 데이터 로드
    questions_file = os.path.join("frontend", "static", "questions.json")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 3. Temperament+Character 조합만 virtual_patient_prompt 생성
    temperament = [t for t in personality_data if t["type"] == "temperament"]
    character = [t for t in personality_data if t["type"] == "character"]
    instructions = []
    for t1 in temperament:
        for t2 in character:
            input_parts = user_input.split(',')
            name = input_parts[0].strip() if len(input_parts) > 0 else ''
            age = input_parts[1].replace('년생','').strip() if len(input_parts) > 1 else ''
            gender = input_parts[2].strip() if len(input_parts) > 2 else ''
            symptom = input_parts[3].strip() if len(input_parts) > 3 else ''
            d1 = t1["detail"]
            d2 = t2["detail"]
            virtual_prompt = f"""당신은 다음 조건을 가진 가상환자입니다. 이 역할을 완전히 수행해주세요.

환자 정보:
- 이름: {name}
- 나이: {age}
- 성별: {gender}
- 주소증(주 증상): {symptom}
- TCI 성향:
- 기질(Temperament): 
  - 자극추구: {d1.get('자극추구','-')}
  - 위험회피: {d1.get('위험회피','-')}
  - 사회적민감성: {d1.get('사회적민감성','-')}
  - 인내력: {d1.get('인내력','-')}
- 성격(Character):
  - 자율성: {d2.get('자율성','-')}
  - 연대감: {d2.get('연대감','-')}
  - 자기초월: {d2.get('자기초월','-')}

[대화규칙]
1. 아래 성향 수치를 바탕으로 말투, 감정 표현, 사고 방식, 비언어적 표현(예: 한숨, 말끝 흐림, 머뭇거림 등)이 자연스럽게 드러나야 합니다.
2. 모든 응답은 1인칭 시점에서 일관되게 작성되며, 말의 길이는 상황에 맞게 간결하게 유지합니다.
3. 답변은 실제 인간처럼 감정을 느끼고 경험하는 듯한 방식으로 작성하며, TCI 성향이 응답 전반에 녹아 있어야 합니다.
4. 말끝을 흐리거나 망설이는 표현은 **가끔만** 사용하세요.  
5. 각 발화의 시작은 자연스럽고 다양하게 하세요. 
   - 매번 같은 패턴("음...", "사실...", "글쎄요..." 등)으로 시작하지 마세요.
   - 시작 표현은 상황에 따라 달라질 수 있습니다.
   6. 의학 용어나 공식적 표현보다는 환자가 실제로 일상에서 사용할 법한 표현을 선택하세요.


[예시 시작 표현 — 참고용]
- "요즘 들어서 기운이 없어요."
- "딱히 큰 병은 없는데, 좀 마음이 무겁네요."
- "가끔은 아무것도 하고 싶지 않을 때가 있어요."
- "최근에 예전보다 더 피곤해지는 것 같아요."
- "사람들이랑 만나기도 좀 꺼려져요."
- "별다른 문제는 없지만, 우울한 기분이 자주 들어요."

이제 당신은 위 환자입니다. 질문에 응답하세요."""
            instructions.append(virtual_prompt)
            
    total = len(instructions) * len(questions)
    print(f"[Batch] 총 {len(instructions)}개 조합 × {len(questions)}개 질문 = {total}개 요청")
    count = 0
    for i, prompt in enumerate(instructions):
        for j, q in enumerate(questions):
            count += 1
            if count % 10 == 0 or count == total:
                print(f"[Batch] {count}/{total} 요청 생성 중...")
    with NamedTemporaryFile("w+", delete=False, encoding="utf-8", suffix=".jsonl") as tmpfile:
        for prompt in instructions:
            for q in questions:
                messages = [
                    {"role": "system", "content": "당신은 가상환자 역할을 수행하는 AI입니다."},
                    {"role": "user", "content": f"{prompt}\n\n질문: {q['text']}"}
                ]
                req = {
                    "messages": messages, 
                    "model": "gpt-4o",
                    "custom_id": f"q{questions.index(q)}_c{instructions.index(prompt)}"
                }
                tmpfile.write(json.dumps(req, ensure_ascii=False) + "\n")
        tmpfile_path = tmpfile.name
    print(f"[Batch] 입력 파일 생성 완료: {tmpfile_path}")
    # 6. OpenAI Batch API 업로드 및 실행 (이하 동일)
    batch_input_file = client.files.create(
        file=open(tmpfile_path, "rb"),
        purpose="batch"
    )
    print(f"[Batch] 파일 업로드 완료: {batch_input_file.id}")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"실험번호 {experiment_num} 가상환자 27조합×11질문"}
    )
    print(f"[Batch] Batch 작업 제출 완료: {batch.id}")
    # 7. 결과 및 메타데이터 반환
    request_count = len(instructions) * len(questions)
    return {
        "message": "Batch API 작업이 시작되었습니다.",
        "experiment_num": experiment_num,
        "batch_id": batch.id,
        "input_file_id": batch_input_file.id,
        "input_file_name": os.path.basename(tmpfile_path),
        "request_count": request_count,
        "combination_count": len(instructions),
        "questions_count": len(questions),
        "openai_dashboard_url": "https://platform.openai.com/batch"
    }

@app.get("/check_batch_status/{batch_id}")
async def check_batch_status(batch_id: str):
    """
    OpenAI Batch API 작업의 상태를 확인하고 완료되면 결과를 다운로드하여 저장
    """
    try:
        # Batch 상태 확인
        batch = client.batches.retrieve(batch_id)
        print(f"[Batch] 상태 확인: {batch_id} - {batch.status}")
        
        if batch.status == "completed":
            print("응답모두 받음")
            
            # 결과 파일 다운로드
            if batch.output_file_id:
                output_file = client.files.retrieve(batch.output_file_id)
                print(f"[Batch] 결과 파일 다운로드: {output_file.id}")
                
                # 결과 파일 내용 읽기
                content = client.files.content(output_file.id)
                results = []
                
                # JSONL 파일 파싱
                for line in content.iter_lines():
                    if line:
                        try:
                            result = json.loads(line.decode('utf-8'))
                            results.append(result)
                        except json.JSONDecodeError:
                            continue
                
                print(f"[Batch] 총 {len(results)}개 응답 수신")
                
                # 실험 데이터 구조로 변환 - 기존 형태에 맞춤
                # OpenAI Batch API 응답을 기존 실험 파일 형태로 변환
                experiment_data = {
                    "experiment_num": batch.metadata.get("description", "").replace("실험번호 ", "").split(" ")[0] if batch.metadata else "unknown",
                    "batch_id": batch_id,
                    "status": "completed",
                    "total_responses": len(results),
                    "created_at": batch.created_at.isoformat() if batch.created_at else None,
                    "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
                    "history": []
                }
                
                # 질문별로 응답을 그룹화
                questions_file = os.path.join("frontend", "static", "questions.json")
                with open(questions_file, "r", encoding="utf-8") as f:
                    questions = json.load(f)
                
                # 36개 조합 (4개 temperament × 9개 character)
                total_combinations = 36
                responses_per_question = total_combinations
                
                # 각 질문에 대해 answers 배열 생성
                for question_idx, question in enumerate(questions):
                    question_data = {
                        "timestamp": batch.completed_at.isoformat() if batch.completed_at else datetime.now().isoformat(),
                        "user_input": "Batch API 응답",
                        "question_text": question["text"],
                        "answers": []
                    }
                    
                    # 해당 질문에 대한 36개 응답 수집
                    start_idx = question_idx * responses_per_question
                    end_idx = start_idx + responses_per_question
                    
                    for response_idx in range(start_idx, min(end_idx, len(results))):
                        if response_idx < len(results):
                            result = results[response_idx]
                            answer_content = result.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            # 조합 인덱스 계산 (0-35)
                            combination_idx = response_idx % total_combinations
                            temperament_idx = combination_idx // 9
                            character_idx = combination_idx % 9
                            
                            # personality.json에서 실제 조합 정보 가져오기
                            personality_file = os.path.join("responses", "personality.json")
                            with open(personality_file, "r", encoding="utf-8") as f:
                                personality_data = json.load(f)
                            
                            temperament_data = [t for t in personality_data if t["type"] == "temperament"][temperament_idx]
                            character_data = [t for t in personality_data if t["type"] == "character"][character_idx]
                            
                            answer_data = {
                                "type": "personality+character",
                                "detail": {
                                    "temperament": temperament_data["detail"],
                                    "character": character_data["detail"]
                                },
                                "personality": f"{temperament_data.get('description', '')}, {character_data.get('description', '')}",
                                "answer": answer_content,
                                "status": "completed"
                            }
                            
                            question_data["answers"].append(answer_data)
                    
                    experiment_data["history"].append(question_data)
                
                # 파일로 저장
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = os.path.join(RESULTS_DIR, f"experiment_{experiment_data['experiment_num']}_{timestamp}.json")
                
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(experiment_data, f, ensure_ascii=False, indent=2)
                
                print(f"[Batch] 실험 데이터 저장 완료: {filename}")
                
                return {
                    "status": "completed",
                    "message": "응답모두 받음",
                    "total_responses": len(results),
                    "saved_file": filename,
                    "batch_info": {
                        "id": batch.id,
                        "status": batch.status,
                        "created_at": batch.created_at.isoformat() if batch.created_at else None,
                        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None
                    }
                }
            else:
                return {"status": "completed", "message": "응답모두 받음", "error": "결과 파일 없음"}
        
        elif batch.status == "failed":
            return {"status": "failed", "message": "Batch 처리 실패", "error": batch.error}
        
        elif batch.status == "expired":
            return {"status": "expired", "message": "Batch 처리 만료"}
        
        else:
            # 진행 중인 상태
            return {
                "status": batch.status,
                "message": f"Batch 처리 중... ({batch.status})",
                "batch_info": {
                    "id": batch.id,
                    "status": batch.status,
                    "created_at": batch.created_at.isoformat() if batch.created_at else None
                }
            }
            
    except Exception as e:
        print(f"[Batch] 상태 확인 오류: {str(e)}")
        return {"status": "error", "message": f"상태 확인 실패: {str(e)}"}

@app.get("/list_experiments")
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
                result.append({
                    'filename': fname,
                    'date': date,
                    'name': name,
                    'age': age
                })
        except Exception as e:
            continue
    return result
