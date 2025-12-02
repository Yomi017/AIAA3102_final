import { apiClient, Message, SessionData } from './api';
import { useState, useEffect } from 'react';

// 事件订阅简单的实现
type Listener = () => void;

class ChatStore {
  sessions: SessionData[] = [];
  currentSessionId: string | null = null;
  currentMessages: Message[] = [];
  loading: boolean = false;
  error: string | null = null;
  systemStatus: {
    status: string;
    model: { loaded: boolean; name: string };
    tools: { total: number; available: string[] };
  } | null = null;

  private listeners: Set<Listener> = new Set();

  subscribe = (listener: Listener) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private notify() {
    this.listeners.forEach((l) => l());
  }

  getSnapshot = () => {
    return {
      sessions: this.sessions,
      currentSessionId: this.currentSessionId,
      currentMessages: this.currentMessages,
      loading: this.loading,
      error: this.error,
      systemStatus: this.systemStatus,
    };
  };

  // ========== Actions ==========

  async checkStatus() {
    try {
      const res = await apiClient.getStatus();
      if (res.code === 200) {
        this.systemStatus = res.data;
      }
    } catch (err) {
      this.systemStatus = null;
    } finally {
      this.notify();
    }
  }

  async loadSessions() {
    this.loading = true;
    this.notify();
    try {
      const res = await apiClient.getHistory();
      if (res.code === 200) {
        this.sessions = res.data.sessions;
      }
    } catch (err: any) {
      this.error = err.message;
    } finally {
      this.loading = false;
      this.notify();
    }
  }

  async selectSession(sessionId: string) {
    if (this.currentSessionId === sessionId) return;
    
    this.currentSessionId = sessionId;
    this.loading = true;
    this.notify();
    try {
      const res = await apiClient.getSession(sessionId);
      if (res.code === 200) {
        this.currentMessages = res.data.messages;
      }
    } catch (err: any) {
      this.error = err.message;
      this.currentMessages = [];
    } finally {
      this.loading = false;
      this.notify();
    }
  }

  async createSession(title?: string) {
    this.loading = true;
    this.notify();
    try {
      const res = await apiClient.createSession(title);
      if (res.code === 200) {
        const newSessionId = res.data.id;
        await this.loadSessions(); // Reload list
        await this.selectSession(newSessionId); // Select it
      }
    } catch (err: any) {
      this.error = err.message;
      this.notify();
    } finally {
      this.loading = false;
      this.notify();
    }
  }

  async deleteSession(sessionId: string) {
    try {
      await apiClient.deleteSession(sessionId);
      if (this.currentSessionId === sessionId) {
        this.currentSessionId = null;
        this.currentMessages = [];
      }
      await this.loadSessions();
    } catch (err: any) {
      this.error = err.message;
      this.notify();
    }
  }

  async sendMessage(content: string, images: string[] = []) {
    if (!this.currentSessionId) return;

    // Optimistic update
    const userMsg: Message = { role: 'user', content };
    this.currentMessages = [...this.currentMessages, userMsg];
    this.notify();

    try {
      // Prepare history for backend (exclude the message we just added locally)
      const historyToSend = this.currentMessages.slice(0, -1).map(m => ({
        role: m.role,
        content: m.content
      }));

      const res = await apiClient.chat({
        message: content,
        session_id: this.currentSessionId,
        history: historyToSend,
        images: images
      });

      if (res.code === 200) {
        // Backend returns updated history including assistant response
        this.currentMessages = res.data.history;
        // Also update session list to show new preview/time
        this.loadSessions(); 
      }
    } catch (err: any) {
      this.error = err.message;
      // Add error message to chat
      this.currentMessages = [...this.currentMessages, { role: 'system', content: `Error: ${err.message}` }];
    } finally {
      this.notify();
    }
  }
  
  resetError() {
    this.error = null;
    this.notify();
  }
}

export const chatStore = new ChatStore();

export function useChatStore() {
  const [state, setState] = useState(chatStore.getSnapshot());

  useEffect(() => {
    const unsubscribe = chatStore.subscribe(() => {
      setState(chatStore.getSnapshot());
    });
    return unsubscribe;
  }, []);

  return state;
}
