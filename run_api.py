import os
import subprocess
import sys

# 确保在正确的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 启动服务器
print("Starting AIAA3102 Agent API server on http://localhost:8000")
print("API Documentation: http://localhost:8000/docs")
print("ReDoc: http://localhost:8000/redoc")
print()

import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )

