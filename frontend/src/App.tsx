import React, { useEffect, useRef } from 'react';
import { chatStore, useChatStore } from './store';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { ToolPanel } from './components/ToolPanel';
import clsx from 'clsx';

function App() {
  const { sessions, currentSessionId, currentMessages, loading, error, systemStatus } = useChatStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatStore.loadSessions();
    chatStore.checkStatus();
    const interval = setInterval(() => {
      chatStore.checkStatus();
    }, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
  }, [currentMessages]);

  const handleSelectSession = (id: string) => {
    chatStore.selectSession(id);
  };

  const handleCreateSession = () => {
    chatStore.createSession();
  };

  const handleDeleteSession = (id: string) => {
    chatStore.deleteSession(id);
  };

  const handleSendMessage = (content: string, images: string[]) => {
    chatStore.sendMessage(content, images);
  };

  // Status Indicator Logic
  const isConnected = !!systemStatus;
  const isModelLoaded = systemStatus?.model?.loaded;
  const statusColor = !isConnected ? 'bg-red-500' : isModelLoaded ? 'bg-green-500' : 'bg-yellow-500';
  const statusText = !isConnected ? 'Disconnected' : isModelLoaded ? 'Online' : 'Initializing...';

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden font-sans">
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onCreateSession={handleCreateSession}
        onDeleteSession={handleDeleteSession}
      />

      <div className="flex-1 flex flex-col min-w-0 relative">
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 p-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold text-gray-800 tracking-tight">
              {currentSessionId 
                ? sessions.find(s => s.id === currentSessionId)?.title || 'Chat'
                : 'AIAA3102 Agent'}
            </h1>
            {loading && (
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
            )}
          </div>
          
          <div className="flex items-center gap-3 text-sm">
             <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 rounded-full border border-gray-200">
                <span className={clsx("w-2.5 h-2.5 rounded-full shadow-sm transition-colors duration-500", statusColor)}></span>
                <span className="text-gray-600 font-medium text-xs uppercase tracking-wider">{statusText}</span>
             </div>
             {systemStatus?.model?.name && (
               <div className="hidden md:block text-xs text-gray-400 font-mono">
                 {systemStatus.model.name}
               </div>
             )}
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth overflow-anchor-none min-h-0">
          <div className="max-w-4xl mx-auto h-full flex flex-col">
            {currentSessionId ? (
              <>
                {currentMessages.length === 0 ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-gray-400 space-y-4">
                    <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center text-3xl">
                      🤖
                    </div>
                    <p className="text-lg font-medium">Start a new conversation</p>
                    <p className="text-sm">Ask anything or use tools to get started.</p>
                  </div>
                ) : (
                  <div className="space-y-6 pb-4">
                    {currentMessages
                      .filter(msg => msg.role !== 'system')
                      .map((msg, idx) => (
                        <ChatMessage key={idx} message={msg} />
                      ))
                    }
                    {loading && (
                      <div className="flex w-full mb-6 justify-start animate-fade-in-up">
                         <div className="w-8 h-8 mr-3 flex-shrink-0 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-sm mt-1">
                            🤖
                         </div>
                         <div className="bg-white border border-gray-100 rounded-2xl rounded-tl-none p-4 shadow-sm flex items-center gap-3">
                            <div className="flex space-x-1.5">
                              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
                              <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                              <div className="w-2 h-2 bg-pink-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                            </div>
                            <span className="text-sm text-gray-500 font-medium animate-pulse">Thinking...</span>
                         </div>
                      </div>
                    )}
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-gray-400 space-y-6">
                 <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl shadow-lg flex items-center justify-center text-4xl text-white">
                    ✨
                 </div>
                 <div className="text-center">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Welcome to Agent Chat</h2>
                    <p className="text-gray-500">Select a session from the sidebar or create a new one.</p>
                 </div>
              </div>
            )}
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="absolute top-20 left-1/2 transform -translate-x-1/2 z-50 animate-fade-in-down">
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <span className="font-medium">{error}</span>
              <button onClick={() => chatStore.resetError()} className="ml-2 text-red-400 hover:text-red-600">&times;</button>
            </div>
          </div>
        )}

        {/* Input Area */}
        {currentSessionId && (
          <div className="p-4 bg-white/80 backdrop-blur border-t border-gray-200">
            <div className="max-w-4xl mx-auto">
              <ChatInput onSendMessage={handleSendMessage} disabled={loading || !isConnected} />
            </div>
          </div>
        )}
      </div>

      <ToolPanel />
    </div>
  );
}

export default App;
