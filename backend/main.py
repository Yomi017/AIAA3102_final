from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
import os
import asyncio
from pathlib import Path
from loguru import logger

# 导入项目中的 Agent 和 Tools
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from llm import Qwen3VL
from tool import ToolsManager

# ==================== 初始化 ====================
app = FastAPI(title="AIAA3102 Agent API", version="1.0.0")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建数据目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
HISTORY_FILE = Path("chat_history.json")

# 全局资源（延迟初始化）
llm: Optional[Qwen3VL] = None
agent: Optional[Agent] = None
sessions: Dict[str, Dict[str, Any]] = {}

# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    images: Optional[List[str]] = None

class ChatResponse(BaseModel):
    """聊天响应"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]
    timestamp: str

class ToolInfo(BaseModel):
    """工具信息"""
    name_for_model: str
    name_for_human: str
    description_for_model: str
    parameters: List[Dict[str, Any]]
    timeout: int

class ToolsResponse(BaseModel):
    """工具列表响应"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]
    timestamp: str

class Session(BaseModel):
    """会话"""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str

class SessionDetail(BaseModel):
    """会话详情"""
    id: str
    title: str
    created_at: str
    messages: List[Dict[str, Any]]

class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    title: Optional[str] = None

class RenameSessionRequest(BaseModel):
    """重命名会话请求"""
    title: str

class ExecuteToolRequest(BaseModel):
    """执行工具请求"""
    args: Dict[str, Any]

class ErrorResponse(BaseModel):
    """错误响应"""
    code: int
    message: str
    error: Optional[Dict[str, Any]] = None
    timestamp: str

# ==================== 应用启动/关闭 ====================

async def initialize_resources():
    """初始化模型和 Agent"""
    global llm, agent
    
    logger.info("Initializing model and agent...")
    try:
        llm = Qwen3VL()
        agent = Agent(llm, rag_db_path="rag/wiki_vector_db" if os.path.exists("rag/wiki_vector_db") else None)
        load_history()
        logger.success("Model and agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        raise

async def shutdown_resources():
    """关闭资源"""
    global llm, agent
    logger.info("Shutting down resources...")
    # 可以在这里添加清理逻辑
    logger.info("Resources shutdown complete")

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    await initialize_resources()

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    await shutdown_resources()

# ==================== 工具函数 ====================

def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().isoformat()

def load_history():
    """从文件加载历史记录"""
    global sessions
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            logger.info(f"Loaded {len(sessions)} sessions from history")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            sessions = {}
    else:
        sessions = {}

def save_history():
    """保存历史记录到文件"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
        logger.info("History saved successfully")
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

def create_session_data(title: Optional[str] = None) -> str:
    """创建新会话"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "id": session_id,
        "title": title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": get_timestamp(),
        "updated_at": get_timestamp(),
        "messages": []
    }
    save_history()
    return session_id

def format_session(session_id: str) -> Session:
    """格式化会话数据"""
    session = sessions[session_id]
    messages = session.get("messages", [])
    preview = ""
    if messages:
        last_msg = messages[-1]
        preview = last_msg.get("content", "")[:100]
    
    return {
        "id": session_id,
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "message_count": len(messages),
        "preview": preview
    }

# ==================== API 路由 ====================

# ========== 聊天接口 ==========

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送聊天消息
    
    - **message**: 用户消息
    - **session_id**: 会话ID（可选，不提供则创建新会话）
    - **history**: 对话历史
    - **images**: 图片列表
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        # 处理会话
        session_id = request.session_id
        if not session_id:
            session_id = create_session_data()
        elif session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        history = request.history or session.get("messages", [])
        
        logger.info(f"Chat request - Session: {session_id}, Message: {request.message[:100]}")
        
        # 调用 Agent
        response, new_history = agent.text(request.message, history=history, images=request.images)
        
        # 保存消息到会话
        session["messages"] = new_history
        session["updated_at"] = get_timestamp()
        save_history()
        
        return ChatResponse(
            data={
                "response": response,
                "session_id": session_id,
                "history": new_history
            },
            timestamp=get_timestamp()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/chat/stream")
async def chat_stream(message: str, session_id: Optional[str] = None, images: Optional[str] = None):
    """
    SSE 流式聊天响应
    
    返回 Server-Sent Events 格式的流式响应
    """
    async def generate():
        try:
            if not agent:
                yield f"data: {json.dumps({'error': 'Agent not initialized'})}\n\n"
                return
            
            # 处理会话
            sid = session_id
            if not sid:
                sid = create_session_data()
            elif sid not in sessions:
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                return
            
            session = sessions[sid]
            history = session.get("messages", [])
            image_list = json.loads(images) if images else None
            
            yield f"data: {json.dumps({'type': 'start', 'session_id': sid})}\n\n"
            
            # 调用 Agent
            response, new_history = agent.text(message, history=history, images=image_list)
            
            # 保存消息
            session["messages"] = new_history
            session["updated_at"] = get_timestamp()
            save_history()
            
            yield f"data: {json.dumps({'type': 'response', 'content': response})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
        
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ========== 工具接口 ==========

@app.get("/api/v1/tools", response_model=ToolsResponse)
async def get_tools():
    """
    获取所有可用工具列表
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        tools_info = agent.tool.get_all_tools_info()
        return ToolsResponse(
            data={
                "tools": tools_info,
                "total": len(tools_info)
            },
            timestamp=get_timestamp()
        )
    except Exception as e:
        logger.error(f"Get tools error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tools/{tool_name}", response_model=ToolsResponse)
async def get_tool(tool_name: str):
    """
    获取单个工具详情
    
    - **tool_name**: 工具名称 (name_for_model)
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        tool = agent.tool._tools.get(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        return ToolsResponse(
            data=tool.get_config(),
            timestamp=get_timestamp()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get tool error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/tools/{tool_name}/execute", response_model=Dict[str, Any])
async def execute_tool(tool_name: str, request: ExecuteToolRequest):
    """
    直接执行工具（调试用）
    
    - **tool_name**: 工具名称
    - **args**: 工具参数
    """
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        logger.info(f"Executing tool: {tool_name} with args: {request.args}")
        result = agent.tool.call_tool(tool_name, **request.args)
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "result": result,
                "execution_time": 0
            },
            "timestamp": get_timestamp()
        }
    except Exception as e:
        logger.error(f"Execute tool error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 历史记录接口 ==========

@app.get("/api/v1/history")
async def get_history(page: int = 1, limit: int = 20):
    """
    获取所有会话列表
    
    - **page**: 页码
    - **limit**: 每页数量
    """
    try:
        all_sessions = list(sessions.items())
        total = len(all_sessions)
        
        # 按更新时间倒序
        sorted_sessions = sorted(
            all_sessions,
            key=lambda x: x[1].get("updated_at", ""),
            reverse=True
        )
        
        # 分页
        start = (page - 1) * limit
        end = start + limit
        paginated = sorted_sessions[start:end]
        
        session_list = [
            format_session(sid)
            for sid, _ in paginated
        ]
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "sessions": session_list,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total
                }
            },
            "timestamp": get_timestamp()
        }
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history/{session_id}")
async def get_session(session_id: str):
    """
    获取单个会话详情
    
    - **session_id**: 会话ID
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        return {
            "code": 200,
            "message": "success",
            "data": {
                "id": session_id,
                "title": session["title"],
                "created_at": session["created_at"],
                "messages": session.get("messages", [])
            },
            "timestamp": get_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/history")
async def create_session(request: CreateSessionRequest):
    """
    创建新会话
    
    - **title**: 会话标题（可选）
    """
    try:
        session_id = create_session_data(request.title)
        session = sessions[session_id]
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "id": session_id,
                "title": session["title"],
                "created_at": session["created_at"]
            },
            "timestamp": get_timestamp()
        }
    except Exception as e:
        logger.error(f"Create session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/history/{session_id}/rename")
async def rename_session(session_id: str, request: RenameSessionRequest):
    """
    重命名会话
    
    - **session_id**: 会话ID
    - **title**: 新标题
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        sessions[session_id]["title"] = request.title
        sessions[session_id]["updated_at"] = get_timestamp()
        save_history()
        
        return {
            "code": 200,
            "message": "success",
            "data": {"status": "renamed"},
            "timestamp": get_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rename session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/history/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话
    
    - **session_id**: 会话ID
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        del sessions[session_id]
        save_history()
        
        return {
            "code": 200,
            "message": "success",
            "data": {"status": "deleted"},
            "timestamp": get_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/history/clear")
async def clear_history(confirm: bool = False):
    """
    清空所有历史记录
    
    - **confirm**: 确认标志（必须为 true）
    """
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="Confirmation required")
        
        sessions.clear()
        save_history()
        
        return {
            "code": 200,
            "message": "success",
            "data": {"status": "cleared"},
            "timestamp": get_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 文件上传接口 ==========

@app.post("/api/v1/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片
    
    - **file**: 图片文件
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only images are allowed")
        
        # 生成文件名
        ext = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = UPLOAD_DIR / filename
        
        # 保存文件
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        logger.info(f"Image uploaded: {filename}")
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "url": f"/uploads/{filename}",
                "path": str(filepath),
                "size": len(content),
                "filename": filename
            },
            "timestamp": get_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/upload/{filename}")
async def delete_upload(filename: str):
    """
    删除上传的文件
    
    - **filename**: 文件名
    """
    try:
        filepath = UPLOAD_DIR / filename
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        filepath.unlink()
        logger.info(f"File deleted: {filename}")
        
        return {
            "code": 200,
            "message": "success",
            "data": {"status": "deleted"},
            "timestamp": get_timestamp()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== 系统接口 ==========

@app.get("/api/v1/status")
async def get_status():
    """获取系统状态"""
    try:
        if not agent:
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "status": "initializing",
                    "model": {
                        "name": "Qwen3VL",
                        "supports_multimodal": True,
                        "loaded": False
                    },
                    "tools": {
                        "total": 0,
                        "available": []
                    },
                    "sessions": len(sessions)
                },
                "timestamp": get_timestamp()
            }
        
        return {
            "code": 200,
            "message": "success",
            "data": {
                "status": "running",
                "model": {
                    "name": "Qwen3VL",
                    "supports_multimodal": agent.supports_multimodal,
                    "loaded": True
                },
                "tools": {
                    "total": len(agent.tool.get_all_tools_info()),
                    "available": list(agent.tool.get_tool_names())
                },
                "sessions": len(sessions)
            },
            "timestamp": get_timestamp()
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/config")
async def get_config():
    """获取前端配置"""
    return {
        "code": 200,
        "message": "success",
        "data": {
            "max_upload_size": 10 * 1024 * 1024,  # 10MB
            "supported_formats": ["jpg", "jpeg", "png", "gif", "webp"],
            "api_version": "v1",
            "supported_multimodal": agent.supports_multimodal if agent else True
        },
        "timestamp": get_timestamp()
    }

# ========== 根路由 ==========

@app.get("/")
async def root():
    """根路由"""
    return {
        "name": "AIAA3102 Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

# ==================== 错误处理 ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": "error",
            "error": {
                "detail": exc.detail
            },
            "timestamp": get_timestamp()
        }
    )

# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AIAA3102 Agent API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
