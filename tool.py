import os
import yaml
from typing import Any, Dict, List, Optional

from langchain_google_community import GoogleSearchAPIWrapper

class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
        self._google_search_wrapper: Optional[GoogleSearchAPIWrapper] = None
    
    def _tools(self) -> list:
        
        tools = [
            {
                'name_for_human': 'Google Search',
                'name_for_model': 'google_search',
                'description_for_model': 'Google Search is a general search engine that can be used to access the internet, query encyclopedic knowledge, and learn about current events.',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': 'Search query or phrase to look up on Google Search.',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
            {
                'name_for_human': '天气查询',
                'name_for_model': 'query_weather',
                'description_for_model': '一个专门用于查询特定城市实时天气的工具，需要同时提供城市和省份的名称。',
                'parameters': [
                    {
                        'name': 'city',
                        'description': '需要查询天气的城市名称，例如“成都”',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'province',
                        'description': '需要查询天气城市所在的省份，例如“四川”',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
            {
                'name_for_human': '时间查询',
                'name_for_model': 'query_time',
                'description_for_model': '一个用于查询当前时间的工具，不需要任何参数。',
                'parameters': []
            }
        ]
        return tools

    