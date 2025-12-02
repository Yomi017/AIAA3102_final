import React, { useState, useRef } from 'react';
import { apiClient } from '../api';

interface ChatInputProps {
  onSendMessage: (content: string, images: string[]) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled }) => {
  const [input, setInput] = useState('');
  // Store objects with id (filename) and url (for preview)
  const [images, setImages] = useState<{id: string, url: string}[]>([]);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    if (!input.trim() && images.length === 0) return;
    // Send only the IDs (filenames) to the backend
    onSendMessage(input, images.map(img => img.id));
    setInput('');
    setImages([]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const uploadFile = async (file: File) => {
    if (!file.type.startsWith('image/')) return;
    setUploading(true);
    try {
      const res = await apiClient.uploadImage(file);
      if (res.code === 200) {
        setImages(prev => [...prev, { id: res.data.filename, url: res.data.url }]);
      }
    } catch (error) {
      console.error('Upload failed', error);
      alert('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await uploadFile(e.target.files[0]);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    // Handle Files
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      for (const file of files) {
        await uploadFile(file);
      }
    }

    // Handle Text
    const text = e.dataTransfer.getData('text');
    if (text) {
      setInput(prev => prev + (prev ? '\n' : '') + text);
    }
  };

  return (
    <div 
      className={`relative transition-all duration-200 ${isDragging ? 'scale-[1.02]' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging && (
        <div className="absolute inset-0 bg-blue-50/90 z-20 rounded-xl flex items-center justify-center border-2 border-dashed border-blue-400 backdrop-blur-sm">
          <div className="text-blue-600 font-medium flex flex-col items-center animate-bounce">
            <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
            <span className="text-lg">Drop text or images here</span>
          </div>
        </div>
      )}
      {images.length > 0 && (
        <div className="flex gap-3 mb-3 overflow-x-auto p-2 bg-gray-50 rounded-lg border border-gray-100">
          {images.map((img, idx) => (
            <div key={idx} className="relative w-20 h-20 flex-shrink-0 group">
              <img src={img.url} alt="uploaded" className="w-full h-full object-cover rounded-lg shadow-sm border border-gray-200" />
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
