import React, { useState, useRef } from 'react';
import { apiClient } from '../api';

interface ChatInputProps {
  onSendMessage: (content: string, images: string[]) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled }) => {
  const [input, setInput] = useState('');
  const [images, setImages] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    if (!input.trim() && images.length === 0) return;
    onSendMessage(input, images);
    setInput('');
    setImages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setUploading(true);
      try {
        const file = e.target.files[0];
        const res = await apiClient.uploadImage(file);
        if (res.code === 200) {
          setImages([...images, res.data.url]);
        }
      } catch (error) {
        console.error('Upload failed', error);
        alert('Upload failed');
      } finally {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="relative">
      {images.length > 0 && (
        <div className="flex gap-3 mb-3 overflow-x-auto p-2 bg-gray-50 rounded-lg border border-gray-100">
          {images.map((img, idx) => (
            <div key={idx} className="relative w-20 h-20 flex-shrink-0 group">
              <img src={img} alt="uploaded" className="w-full h-full object-cover rounded-lg shadow-sm border border-gray-200" />
              <button
                onClick={() => setImages(images.filter((_, i) => i !== idx))}
                className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs shadow-md opacity-0 group-hover:opacity-100 transition-opacity"
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      )}
      
      <div className="flex gap-3 items-end bg-white rounded-xl border border-gray-200 shadow-sm p-2 focus-within:ring-2 focus-within:ring-blue-100 focus-within:border-blue-400 transition-all">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || uploading}
          className="p-2.5 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
          title="Upload Image"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
        </button>
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          accept="image/*"
          onChange={handleFileChange}
        />
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={disabled}
          className="flex-1 p-2.5 bg-transparent border-none focus:ring-0 resize-none max-h-32 min-h-[44px] text-gray-700 placeholder-gray-400"
          rows={1}
          style={{ height: 'auto', minHeight: '44px' }}
        />
        <button
          onClick={handleSend}
          disabled={disabled || (!input.trim() && images.length === 0)}
          className="p-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors shadow-sm"
        >
          <svg className="w-5 h-5 transform rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
        </button>
      </div>
      <div className="text-center mt-2 text-xs text-gray-400">
        Press Enter to send, Shift + Enter for new line
      </div>
    </div>
  );
};
