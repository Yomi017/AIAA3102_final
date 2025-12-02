import axios, { AxiosResponse } from 'axios';

// ==================== 类型定义 ====================

export interface Message {
  role: string;
  content: any;
  // 允许其他字段，但主要使用这两个
  [key: string]: any;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  history?: Message[];
  images?: string[];
}

export interface ChatResponseData {
  response: string;
  session_id: string;
  history: Message[];
}

export interface ChatResponse {
  code: int;
  message: string;
  data: ChatResponseData;
  timestamp: string;
}

export interface SessionData {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: int;
  preview: string;
}

export interface SessionDetail {
  id: string;
  title: string;
  created_at: string;
  messages: Message[];
}

export interface HistoryResponse {
  code: int;
  message: string;
  data: {
    sessions: SessionData[];
    pagination: {
      page: int;
      limit: int;
      total: int;
    };
  };
  timestamp: string;
}

export interface ToolParameter {
  name: string;
  type: string;
  description: string;
  required: boolean;
}

export interface Tool {
  name_for_model: string;
  name_for_human: string;
  description_for_model: string;
  parameters: ToolParameter[];
  timeout: number;
}

export interface ToolsResponse {
  code: int;
  message: string;
  data: {
    tools: Tool[];
    total: int;
  };
  timestamp: string;
}

export interface UploadResponse {
  code: int;
  message: string;
  data: {
    url: string;
    path: string;
    size: int;
    filename: string;
  };
  timestamp: string;
}

type int = number;

// ==================== API 客户端 ====================

class APIClient {
  private client;

  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1', // 使用相对路径，通过 Vite 代理转发
      timeout: 600000, // 10 minutes
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => response.data,
      (error) => {
        console.error('API Error:', error);
        if (error.response) {
          // 服务器返回了错误状态码
          return Promise.reject(new Error(error.response.data.detail || error.response.data.message || 'Server Error'));
        } else if (error.request) {
          // 请求已发出但没有收到响应
          return Promise.reject(new Error('No response from server'));
        } else {
          // 请求配置出错
          return Promise.reject(new Error(error.message));
        }
      }
    );
  }

  // 聊天
  async chat(data: ChatRequest): Promise<ChatResponse> {
    return this.client.post('/chat', data);
  }

  // 获取历史会话列表
  async getHistory(page: number = 1, limit: number = 20): Promise<HistoryResponse> {
    return this.client.get('/history', { params: { page, limit } });
  }

  // 获取单个会话详情
  async getSession(sessionId: string): Promise<{ code: number; message: string; data: SessionDetail; timestamp: string }> {
    return this.client.get(`/history/${sessionId}`);
  }

  // 创建新会话
  async createSession(title?: string): Promise<{ code: number; message: string; data: { id: string; title: string; created_at: string }; timestamp: string }> {
    return this.client.post('/history', { title });
  }

  // 重命名会话
  async renameSession(sessionId: string, title: string): Promise<any> {
    return this.client.post(`/history/${sessionId}/rename`, { title });
  }

  // 删除会话
  async deleteSession(sessionId: string): Promise<any> {
    return this.client.delete(`/history/${sessionId}`);
  }

  // 清空历史
  async clearHistory(): Promise<any> {
    return this.client.post('/history/clear', {}, { params: { confirm: true } });
  }

  // 获取工具列表
  async getTools(): Promise<ToolsResponse> {
    return this.client.get('/tools');
  }

  // 上传图片
  async uploadImage(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.client.post('/upload/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  // 获取系统状态
  async getStatus(): Promise<any> {
    return this.client.get('/status');
  }
}

export const apiClient = new APIClient();
