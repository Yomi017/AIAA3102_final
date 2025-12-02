import React, { useEffect, useState } from 'react';
import { apiClient, Tool } from '../api';

export const ToolPanel: React.FC = () => {
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadTools = async () => {
      setLoading(true);
      try {
        const res = await apiClient.getTools();
        if (res.code === 200) {
          setTools(res.data.tools);
        }
      } catch (error) {
        console.error('Failed to load tools', error);
      } finally {
        setLoading(false);
      }
    };
    loadTools();
  }, []);

  if (loading) return <div className="p-4 text-gray-500">Loading tools...</div>;

  return (
    <div className="w-64 bg-gray-50 border-l h-full overflow-y-auto p-4 hidden lg:block">
      <h3 className="font-bold mb-4 text-gray-700">Available Tools</h3>
      <div className="space-y-3">
        {tools.map((tool) => (
          <div key={tool.name_for_model} className="bg-white p-3 rounded border shadow-sm">
            <div className="font-medium text-sm text-gray-900">{tool.name_for_human}</div>
            <div className="text-xs text-gray-500 mt-1">{tool.description_for_model}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
