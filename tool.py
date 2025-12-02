import os
import yaml
import json
import requests
from typing import Any, Dict, List, Optional
from loguru import logger

from langchain_google_community import GoogleSearchAPIWrapper
from calculator import Calculator
from rag.rag_engine import RAGEngine

class Tools:
    def __init__(self, rag_db_path: Optional[str] = None) -> None:
        self.toolConfig = self._tools()
        self._google_search_wrapper: Optional[GoogleSearchAPIWrapper] = None
        self.calc = Calculator()
        
        # 初始化RAG引擎(如果提供了数据库路径)
        self.rag_engine = None
        if rag_db_path and os.path.exists(rag_db_path):
            try:
                self.rag_engine = RAGEngine(rag_db_path)
                logger.info(f"RAG engine initialized with database: {rag_db_path}")
            except Exception as e:
                logger.error(f"RAG engine initialization failed: {e}")
        logger.info(f"Tools initialized with {len(self.toolConfig)} tools")
    
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
                'description_for_model': '一个专门用于查询特定城市实时天气的工具，需要同时提供城市和省份的名称',
                'parameters': [
                    {
                        'name': 'city',
                        'description': '需要查询天气的城市名称，例如"成都"',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'province',
                        'description': '需要查询天气城市所在的省份，例如"四川"',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
            {
                'name_for_human': '时间查询',
                'name_for_model': 'query_time',
                'description_for_model': '一个用于查询当前时间的工具，不需要任何参数',
                'parameters': []
            },
            {
                'name_for_human': '基本计算器',
                'name_for_model': 'basic_calculator',
                'description_for_model': '用于基本数学运算的计算器,支持加减乘除、幂运算、平方根、绝对值、指数、对数等。可以计算数学表达式',
                'parameters': [
                    {
                        'name': 'expression',
                        'description': '要计算的数学表达式,例如"2+3*4"、"sqrt(16)"、"abs(-5)"、"2**3"等。支持运算符:+,-,*,/,**。支持函数:sqrt(平方根),abs(绝对值),exp(指数),log(对数)。支持常数:pi,e',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ]
            },
            {
                'name_for_human': '三角函数计算器',
                'name_for_model': 'trig_calculator',
                'description_for_model': '用于计算三角函数和反三角函数。支持sin(正弦)、cos(余弦)、tan(正切)、asin(反正弦)、acos(反余弦)、atan(反正切), 默认输入为弧度制, 若要角度制, 请设置 degree=True',
                'parameters': [
                    {
                        'name': 'function',
                        'description': '要计算的三角函数名称,可选值:"sin","cos","tan","asin","acos","atan"',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'x',
                        'description': '函数的输入值。对于sin/cos/tan,默认使用弧度制;对于asin/acos,输入值必须在[-1,1]范围内',
                        'required': True,
                        'schema': {'type': 'number'},
                    },
                    {
                        'name': 'degree',
                        'description': '是否使用角度制(默认False使用弧度制)。仅对sin/cos/tan有效',
                        'required': False,
                        'schema': {'type': 'boolean'},
                    }
                ]
            },
            {
                'name_for_human': '矩阵计算器',
                'name_for_model': 'matrix_calculator',
                'description_for_model': '用于矩阵运算,支持矩阵加法、减法、乘法、数乘、求行列式、求逆矩阵',
                'parameters': [
                    {
                        'name': 'operation',
                        'description': '要执行的矩阵运算,可选值:"add"(加法),"subtract"(减法),"multiply"(乘法),"scalar_multiply"(数乘),"determinant"(行列式),"inverse"(逆矩阵)',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'matrix_a',
                        'description': '第一个矩阵,格式为嵌套列表,例如[[1,2],[3,4]]',
                        'required': True,
                        'schema': {'type': 'array'},
                    },
                    {
                        'name': 'matrix_b',
                        'description': '第二个矩阵(仅用于add/subtract/multiply操作),格式为嵌套列表',
                        'required': False,
                        'schema': {'type': 'array'},
                    },
                    {
                        'name': 'scalar',
                        'description': '标量值(仅用于scalar_multiply操作)',
                        'required': False,
                        'schema': {'type': 'number'},
                    }
                ]
            },
            {
                'name_for_human': '定积分计算器',
                'name_for_model': 'integral_calculator',
                'description_for_model': '用于计算定积分。可以计算各种函数在指定区间上的定积分',
                'parameters': [
                    {
                        'name': 'func_str',
                        'description': '被积函数的字符串表示,使用x作为变量。例如"x**2"(x的平方),"np.sin(x)"(sin(x)),"x**3 + 2*x"等',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'a',
                        'description': '积分下限',
                        'required': True,
                        'schema': {'type': 'number'},
                    },
                    {
                        'name': 'b',
                        'description': '积分上限',
                        'required': True,
                        'schema': {'type': 'number'},
                    }
                ]
            },
            {
                'name_for_human': '知识库问答',
                'name_for_model': 'knowledge_base_query',
                'description_for_model': '一个基于向量检索的知识库问答工具,可以从预先构建的文档库中检索相关信息来回答问题。当用户询问特定领域知识、文档内容、或需要基于已有资料回答时使用此工具。如果使用网络查询失败,可以尝试使用此工具从知识库中获取答案。',
                'parameters': [
                    {
                        'name': 'question',
                        'description': '用户的问题,需要从知识库中检索答案',
                        'required': True,
                        'schema': {'type': 'string'},
                    },
                    {
                        'name': 'top_k',
                        'description': '返回最相关的文档数量,默认为3',
                        'required': False,
                        'schema': {'type': 'integer'},
                    }
                ]
            }
        ]
        return tools
    
    def google_search(self, search_query: str) -> str:
        """执行谷歌搜索"""
        logger.info(f"Google search: {search_query}")
        
        try:
            url = "http://www.gpts-cristiano.com/cristiano/googleApi"
            payload = json.dumps({"q": search_query})
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            response = requests.post(url, headers=headers, data=payload).json()
            logger.success(f"Google search completed successfully")
            return response['organic'][0]['snippet']
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            raise

    def query_weather(self, city: str, province: str) -> str:
        """查询天气"""
        logger.info(f"Query weather: {province} {city}")
        
        mock_response = {
            "city": city,
            "province": province,
            "temperature": "28°C",
            "weather": "多云",
            "humidity": "65%",
            "wind_direction": "东南风",
            "wind_power": "3级"
        }
        
        return (
            f"地点：{mock_response['province']}{mock_response['city']}，"
            f"天气：{mock_response['weather']}，"
            f"温度：{mock_response['temperature']}，"
            f"湿度：{mock_response['humidity']}，"
            f"风向：{mock_response['wind_direction']}，"
            f"风力：{mock_response['wind_power']}"
        )

    def query_time(self) -> str:
        """查询当前时间"""
        logger.info("Query time called")
        
        url = "https://api.uuni.cn//api/time"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # 更详细的字段检查
            if "date" in data and "weekday" in data:
                return f"当前时间是：{data['date']}，{data['weekday']}"
            else:
                return "无法获取当前时间，API返回格式异常"
        except Exception as e:
            return f"查询时间时发生错误：{e}"
    
    def basic_calculator(self, expression: str) -> str:
        """基本计算器"""
        logger.info(f"Basic calculator: {expression}")
        
        try:
            result = self.calc.evaluate(expression)
            logger.success(f"Calculation result: {result}")
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return f"计算失败: {str(e)}"
    
    def trig_calculator(self, function: str, x: float, degree: bool = False) -> str:
        """三角函数计算器"""
        
        try:
            func_map = {
                'sin': self.calc.sin,
                'cos': self.calc.cos,
                'tan': self.calc.tan,
                'asin': self.calc.asin,
                'acos': self.calc.acos,
                'atan': self.calc.atan
            }
            
            if function not in func_map:
                return f"不支持的三角函数: {function}"
            
            result = func_map[function](x, degree=degree)
            return f"计算结果: {function}({x}) = {result}"
        except Exception as e:
            return f"计算失败: {str(e)}"
    
    def matrix_calculator(self, operation: str, matrix_a: List[List[float]], 
                         matrix_b: List[List[float]] = None, scalar: float = None) -> str:
        """矩阵计算器"""
        
        try:
            if operation == 'add':
                result = self.calc.matrix_add(matrix_a, matrix_b)
                return f"矩阵加法结果: {result}"
            elif operation == 'subtract':
                result = self.calc.matrix_subtract(matrix_a, matrix_b)
                return f"矩阵减法结果: {result}"
            elif operation == 'multiply':
                result = self.calc.matrix_multiply(matrix_a, matrix_b)
                return f"矩阵乘法结果: {result}"
            elif operation == 'scalar_multiply':
                result = self.calc.matrix_scalar_multiply(matrix_a, scalar)
                return f"矩阵数乘结果: {result}"
            elif operation == 'determinant':
                result = self.calc.matrix_determinant(matrix_a)
                return f"矩阵行列式: {result}"
            elif operation == 'inverse':
                result = self.calc.matrix_inverse(matrix_a)
                return f"逆矩阵: {result}"
            else:
                return f"不支持的矩阵运算: {operation}"
        except Exception as e:
            return f"计算失败: {str(e)}"
    
    def integral_calculator(self, func_str: str, a: float, b: float) -> str:
        """定积分计算器"""
        
        try:
            result = self.calc.integrate_function(func_str, a, b)
            return result['description']
        except Exception as e:
            return f"积分计算失败: {str(e)}"
    
    def knowledge_base_query(self, question: str, top_k: int = 3) -> str:
        """知识库问答"""
        logger.info(f"Knowledge base query: {question[:100]}...")
        
        if self.rag_engine is None:
            logger.warning("RAG engine not initialized")
            return "知识库未加载,无法回答问题。请先构建并加载向量数据库。"
        
        try:
            result = self.rag_engine.query(question, top_k=top_k)
            context = result['context']
            logger.success(f"Retrieved {len(result['search_results'])} results from knowledge base")
            
            # 返回检索到的上下文
            return f"根据知识库检索到以下相关信息:\n\n{context}"
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            return f"知识库查询失败: {str(e)}"