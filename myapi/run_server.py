#!/usr/bin/env python3
"""
FastAPI 서버 실행 스크립트
"""

import uvicorn
from main import app

if __name__ == "__main__":
    print("🚀 FastAPI 서버를 시작합니다...")
    print("📍 서버 주소: http://127.0.0.1:8000")
    print("📊 API 문서: http://127.0.0.1:8000/docs")
    print("🧪 테스트 엔드포인트: http://127.0.0.1:8000/test")
    print("⏹️  서버 중지: Ctrl+C")
    print("-" * 50)
    
    try:
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 서버가 중지되었습니다.")
    except Exception as e:
        print(f"❌ 서버 실행 중 오류 발생: {e}")
