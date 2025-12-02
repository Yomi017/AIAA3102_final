import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Message } from '../api';
import clsx from 'clsx';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const isTool = message.role === 'tool';

  const renderContent = (content: any) => {
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      return content.map((item) => {
        if (typeof item === 'string') return item;
        if (item.type === 'text') return item.text || '';
        if (item.type === 'image' || item.image) return `![image](${item.image || item.url || ''})`;
        return '';
      }).join('\n');
    }
    if (typeof content === 'object' && content !== null) {
      if (content.type === 'text') return content.text || '';
      return JSON.stringify(content);
    }
    return String(content);
  };

  const contentStr = renderContent(message.content);

  // Tool Message Rendering
  if (isTool) {
    return (
      <div className="flex w-full mb-2 justify-start pl-10">
        <div className="max-w-[90%] w-full">
          <details className="group border border-purple-200 rounded-lg bg-purple-50/50 overflow-hidden transition-all duration-200 hover:shadow-sm hover:border-purple-300">
            <summary className="cursor-pointer p-2.5 text-sm text-purple-700 font-medium select-none flex items-center gap-2 hover:bg-purple-100/50 transition-colors outline-none">
              <span className="flex items-center justify-center w-5 h-5 rounded bg-purple-100 text-purple-600 text-xs">🛠️</span>
              <span className="flex-1 font-mono text-xs opacity-80">Tool Output: <span className="font-bold text-purple-800">{message.name || 'Unknown Tool'}</span></span>
              <svg className="w-4 h-4 text-purple-400 transition-transform group-open:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
            </summary>
            <div className="p-3 border-t border-purple-100 text-xs font-mono text-gray-600 bg-white/50 whitespace-pre-wrap overflow-x-auto max-h-60 custom-scrollbar">
              {contentStr}
            </div>
          </details>
        </div>
      </div>
    );
  }

  // 解析助手消息，分离思考过程和最终回答
  const parseAssistantMessage = (text: string) => {
    let thinking = '';
    let finalAnswer = '';
    let remaining = text;

    // 1. 处理 </think> 标签
    if (remaining.includes('</think>')) {
      const parts = remaining.split('</think>');
      thinking = parts[0].replace('<think>', '').trim();
      remaining = parts[1];
    }

    // 2. 处理 Final Answer: 标记
    const finalAnswerMarker = 'Final Answer:';
    const markerIndex = remaining.indexOf(finalAnswerMarker);
    
    if (markerIndex !== -1) {
      const processPart = remaining.substring(0, markerIndex).trim();
      finalAnswer = remaining.substring(markerIndex + finalAnswerMarker.length).trim();
      
      // 如果之前没有提取到 thinking，或者有额外的 processPart，合并它们
      if (processPart) {
        thinking = thinking ? `${thinking}\n\n${processPart}` : processPart;
      }
    } else {
      // 如果没有 Final Answer 标记
      // 检查是否包含 React 关键字 (Thought/Action)
      if (remaining.includes('Thought:') || remaining.includes('Action:')) {
        // 认为是纯思考过程
        thinking = thinking ? `${thinking}\n\n${remaining}` : remaining;
      } else {
        // 认为是纯回答
        finalAnswer = remaining.trim();
      }
    }

    return { thinking, finalAnswer };
  };

  const { thinking, finalAnswer } = !isUser && !isSystem 
    ? parseAssistantMessage(contentStr) 
    : { thinking: '', finalAnswer: contentStr };

  // 如果是助手消息且没有最终回答，不显示（除非有思考过程，但用户要求只直接展示 Final Response）
  if (!isUser && !isSystem && !finalAnswer && !thinking) return null;

  // Check for Action in thinking to update summary
  const actionMatch = thinking.match(/Action:\s*([^\n]+)/);
  const actionToolName = actionMatch ? actionMatch[1].trim() : null;
  const summaryText = actionToolName ? `Thinking & Calling: ${actionToolName}` : "Thinking Process";

  return (
    <div className={clsx(
      "flex w-full mb-6 group",
      isUser ? "justify-end" : "justify-start"
    )}>
      {/* Avatar */}
      {!isUser && (
        <div className="w-8 h-8 mr-3 flex-shrink-0 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-sm mt-1">
          {isSystem ? '⚙️' : '🤖'}
        </div>
      )}

      <div className={clsx(
        "max-w-[85%] rounded-2xl p-4 shadow-sm transition-all duration-200",
        isUser 
          ? "bg-blue-600 text-white rounded-tr-none hover:shadow-md" 
          : isSystem 
            ? "bg-red-50 text-red-800 border border-red-100 rounded-tl-none" 
            : "bg-white text-gray-800 border border-gray-100 rounded-tl-none hover:shadow-md"
      )}>
        <div className="prose prose-sm max-w-none dark:prose-invert break-words leading-relaxed">
          {isUser ? (
            <ReactMarkdown 
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                p: ({node, ...props}) => <p className="whitespace-pre-wrap" {...props} />,
                img: ({node, ...props}) => <img className="max-w-full rounded-lg my-2" {...props} />
              }}
            >
              {contentStr}
            </ReactMarkdown>
          ) : (
            <>
              {thinking && (
                <details className={clsx(
                  "mb-3 group/think border rounded-lg overflow-hidden",
                  actionToolName ? "border-blue-200 bg-blue-50/50" : "border-gray-200 bg-gray-50/50"
                )}>
                  <summary className={clsx(
                    "cursor-pointer p-2 text-xs font-medium select-none flex items-center gap-2 hover:bg-opacity-80 transition-colors outline-none",
                    actionToolName ? "text-blue-600 hover:bg-blue-100" : "text-gray-500 hover:bg-gray-100"
                  )}>
                    <span className="w-4 h-4 flex items-center justify-center">{actionToolName ? '🔧' : '💭'}</span>
                    <span className="flex-1">{summaryText}</span>
                    <svg className="w-3 h-3 opacity-50 transition-transform group-open/think:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                  </summary>
                  <div className="p-3 text-xs text-gray-600 whitespace-pre-wrap border-t border-gray-200 bg-white/50 font-mono">
                    <ReactMarkdown 
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {thinking}
                    </ReactMarkdown>
                  </div>
                </details>
              )}
              {finalAnswer && (
                <div className="markdown-body">
                  <ReactMarkdown 
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={{
                      p: ({node, ...props}) => <p className="whitespace-pre-wrap" {...props} />,
                      img: ({node, ...props}) => <img className="max-w-full rounded-lg my-2" {...props} />
                    }}
                  >
                    {finalAnswer}
                  </ReactMarkdown>
                </div>
              )}
              {!finalAnswer && !thinking && <span className="text-gray-400 italic">Empty response</span>}
            </>
          )}
        </div>
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 ml-3 flex-shrink-0 rounded-full bg-gray-200 flex items-center justify-center text-gray-500 shadow-sm mt-1 overflow-hidden">
          👤
        </div>
      )}
    </div>
  );
};
