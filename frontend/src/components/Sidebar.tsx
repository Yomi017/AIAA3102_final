import React from 'react';
import { SessionData } from '../api';
import clsx from 'clsx';

interface SidebarProps {
  sessions: SessionData[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (id: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  currentSessionId,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
}) => {
  return (
    <div className="w-72 bg-gray-900 text-gray-300 flex flex-col h-full border-r border-gray-800 shadow-xl z-20">
      <div className="p-4 border-b border-gray-800">
        <button
          onClick={onCreateSession}
          className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-500 text-white rounded-lg flex items-center justify-center gap-2 transition-all shadow-lg hover:shadow-blue-500/20 font-medium"
        >
          <span className="text-xl leading-none">+</span> New Chat
        </button>
      </div>
      
      <div className="flex-1 overflow-y-auto p-2 space-y-1 custom-scrollbar">
        <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
          History
        </div>
        {sessions.map((session) => (
          <div
            key={session.id}
            className={clsx(
              "group flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all duration-200 border border-transparent",
              currentSessionId === session.id 
                ? "bg-gray-800 text-white border-gray-700 shadow-sm" 
                : "hover:bg-gray-800/50 hover:text-gray-200"
            )}
            onClick={() => onSelectSession(session.id)}
          >
            <div className="flex-1 min-w-0 flex flex-col gap-0.5">
              <div className="truncate text-sm font-medium">{session.title}</div>
              <div className="truncate text-xs text-gray-500 group-hover:text-gray-400">
                {new Date(session.updated_at).toLocaleDateString()}
              </div>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm('Delete this chat?')) {
                  onDeleteSession(session.id);
                }
              }}
              className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 p-1.5 rounded hover:bg-gray-700 transition-all"
              title="Delete chat"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
            </button>
          </div>
        ))}
        
        {sessions.length === 0 && (
          <div className="text-center py-8 text-gray-600 text-sm">
            No history yet
          </div>
        )}
      </div>
      
      <div className="p-4 border-t border-gray-800 text-xs text-gray-600 text-center">
        AIAA3102 Agent v1.0
      </div>
    </div>
  );
};
