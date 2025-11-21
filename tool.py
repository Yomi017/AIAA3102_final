import os
import yaml
import json
import requests
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
    
    def google_search(self, search_query: str) -> str:
        """
        执行谷歌搜索。
        Args:
            search_query (str): 搜索的关键词或短语。
        Returns:
            str: 搜索结果的摘要。
        """
        url = "http://www.gpts-cristiano.com/cristiano/googleApi"

        # 构造请求体
        payload = json.dumps({"q": search_query})
        # 构造请求头，需要填入自己的API KEY
        headers = {
            # 'X-API-KEY': '修改为你自己的key',  # 请替换为你的Serper API密钥
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        # 发送POST请求
        response = requests.post(url, headers=headers, data=payload).json()

        print(response)
        # 返回第一条搜索结果的摘要
        return response['organic'][0]['snippet']

    # ================= 新增的天气查询工具实现 =================
    def query_weather(self, city: str, province: str) -> str:
        """
        查询指定省份和城市的天气。
        Args:
            city (str): 城市名称。
            province (str): 省份名称。
        Returns:
            str: 格式化后的天气信息字符串或错误提示。
        """
        mock_response = {
            "city": city,
            "province": province,
            "temperature": "28°C",
            "weather": "多云",
            "humidity": "65%",
            "wind_direction": "东南风",
            "wind_power": "3级"
        }
        
        # 将字典格式化成一个对LLM友好的字符串
        return (
            f"地点：{mock_response['province']}{mock_response['city']}，"
            f"天气：{mock_response['weather']}，"
            f"温度：{mock_response['temperature']}，"
            f"湿度：{mock_response['humidity']}，"
            f"风向：{mock_response['wind_direction']}，"
            f"风力：{mock_response['wind_power']}"
        )
    # =======================================================
    def query_time(self) -> str:
        """
        查询当前的时间（通过外部API）。
        Returns:
            str: 当前时间的字符串表示。
        """
        url = "https://api.uuni.cn//api/time"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if "date" in data and "weekday" in data:
                return f"当前时间是：{data['date']}，{data['weekday']}"
            else:
                return "无法获取当前时间，API返回格式异常。"
        except Exception as e:
            return f"查询时间时发生错误：{e}"
    