import os
import json
import requests
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from loguru import logger

from langchain_google_community import GoogleSearchAPIWrapper
from calculator import Calculator
from rag.rag_engine import RAGEngine

# 尝试导入 Tavily
try:
    from tavily import TavilyClient
    HAS_TAVILY = True
except ImportError:
    HAS_TAVILY = False
    logger.warning("Tavily not installed. Install with: pip install tavily-python")


class ToolBase(ABC):
    """工具基类"""
    
    # 工具执行默认超时时间（秒），子类可以覆盖
    TIMEOUT = 30
    
    @property
    @abstractmethod
    def name_for_human(self) -> str:
        """人类可读的工具名称"""
        pass
    
    @property
    @abstractmethod
    def name_for_model(self) -> str:
        """模型调用的工具名称"""
        pass
    
    @property
    @abstractmethod
    def description_for_model(self) -> str:
        """工具功能描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[Dict[str, Any]]:
        """工具参数描述"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具"""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """获取工具配置信息"""
        return {
            'name_for_human': self.name_for_human,
            'name_for_model': self.name_for_model,
            'description_for_model': self.description_for_model,
            'parameters': self.parameters
        }
    
    def get_timeout(self) -> int:
        """获取工具执行超时时间（秒）"""
        return self.TIMEOUT


class GoogleSearchTool(ToolBase):
    """Google搜索工具"""
    
    TIMEOUT = 15  # Google搜索超时15秒
    
    def __init__(self, credentials_path: str = None):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_SEARCH_CREDENTIALS", "jsons/goole_search.json")
        self._config: Optional[Dict[str, str]] = None
    
    @property
    def name_for_human(self) -> str:
        return 'Google Search'
    
    @property
    def name_for_model(self) -> str:
        return 'google_search'
    
    @property
    def description_for_model(self) -> str:
        return 'Google Search is a general search engine that can be used to access the internet, query encyclopedic knowledge, and learn about current events.'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'search_query',
                'description': 'Search query or phrase to look up on Google Search.',
                'required': True,
                'schema': {'type': 'string'},
            }
        ]
    
    def _load_config(self) -> Dict[str, str]:
        if self._config is None:
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(f"未找到 Google 配置文件: {self.credentials_path}")
            with open(self.credentials_path, "r", encoding="utf-8") as fp:
                self._config = json.load(fp) or {}
        return self._config
    
    def _search_request(self, query: str, api_key: str, cx: str) -> str:
        endpoint = "https://customsearch.googleapis.com/customsearch/v1"
        params = {"q": query, "cx": cx, "key": api_key, "num": 3}
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            logger.warning("Google search returned no results")
            return "Google 搜索没有找到相关结果。"
        formatted_results = []
        for idx, item in enumerate(items[:3], start=1):
            title = item.get("title", "无标题")
            snippet = item.get("snippet", "无摘要")
            link = item.get("link", "")
            formatted_results.append(f"{idx}. {title}\n{snippet}\n{link}".strip())
        return "\n\n".join(formatted_results)
    
    def execute(self, search_query: str) -> str:
        logger.info(f"Google search: {search_query}")
        try:
            config = self._load_config()
            api_key = config.get("api_key")
            search_engine_id = config.get("search_engine_id")
            if not api_key or not search_engine_id:
                message = "Google 搜索未配置: 请在 JSON 文件中提供 api_key 与 search_engine_id。"
                logger.error(message)
                return message
            
            result = self._search_request(search_query, api_key, search_engine_id)
            logger.success("Google search completed successfully")
            return result
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return f"Google 搜索失败: {str(e)}"


class TavilySearchTool(ToolBase):
    """Tavily 搜索工具 - 更高效的网络搜索"""
    
    TIMEOUT = 15  # Tavily搜索超时15秒
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "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")
        self.client = None
        if HAS_TAVILY and self.api_key:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Tavily client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily client: {e}")
    
    @property
    def name_for_human(self) -> str:
        return 'Tavily Search'
    
    @property
    def name_for_model(self) -> str:
        return 'tavily_search'
    
    @property
    def description_for_model(self) -> str:
        return 'A more efficient web search engine powered by Tavily API. Use this for real-time information, news, current events, and web content search. Returns high-quality, relevant results.'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'search_query',
                'description': 'The search query to look up on the web using Tavily Search.',
                'required': True,
                'schema': {'type': 'string'},
            },
            {
                'name': 'topic',
                'description': '搜索主题，可选值: "general"(通用), "news"(新闻)。默认为"general"',
                'required': False,
                'schema': {'type': 'string'},
            }
        ]
    
    def execute(self, search_query: str, topic: str = "general") -> str:
        logger.info(f"Tavily search: {search_query} (topic: {topic})")
        
        if not HAS_TAVILY:
            message = "Tavily 未安装: 请运行 pip install tavily-python 来安装"
            logger.error(message)
            return message
        
        if not self.client:
            message = "Tavily API key 未配置: 请设置 TAVILY_API_KEY 环境变量"
            logger.error(message)
            return message
        
        try:
            response = self.client.search(
                query=search_query,
                topic=topic,
                include_answer=True,
                max_results=5
            )
            
            if not response or 'results' not in response:
                logger.warning("Tavily search returned no results")
                return "未找到相关搜索结果"
            
            # 格式化结果
            results = response.get('results', [])
            answer = response.get('answer', '')
            
            formatted_results = []
            
            if answer:
                formatted_results.append(f"直接答案: {answer}\n")
            
            for idx, result in enumerate(results[:5], start=1):
                title = result.get('title', '无标题')
                content = result.get('content', '无内容')
                url = result.get('url', '')
                
                formatted_results.append(
                    f"{idx}. {title}\n"
                    f"   {content[:200]}...\n"
                    f"   来源: {url}"
                )
            
            result_text = "\n\n".join(formatted_results)
            logger.success("Tavily search completed successfully")
            return result_text
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return f"Tavily 搜索失败: {str(e)}"


class WeatherQueryTool(ToolBase):
    """天气查询工具"""
    
    TIMEOUT = 10  # 天气查询超时10秒
    
    @property
    def name_for_human(self) -> str:
        return '天气查询'
    
    @property
    def name_for_model(self) -> str:
        return 'query_weather'
    
    @property
    def description_for_model(self) -> str:
        return '一个专门用于查询特定城市实时天气的工具，需要同时提供城市和省份的名称'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
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
        ]
    
    def execute(self, city: str, province: str) -> str:
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


class TimeQueryTool(ToolBase):
    """时间查询工具"""
    
    TIMEOUT = 5  # 时间查询超时5秒
    
    @property
    def name_for_human(self) -> str:
        return '时间查询'
    
    @property
    def name_for_model(self) -> str:
        return 'query_time'
    
    @property
    def description_for_model(self) -> str:
        return '一个用于查询当前时间的工具，不需要任何参数'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return []
    
    def execute(self) -> str:
        logger.info("Query time called")
        
        url = "https://api.uuni.cn//api/time"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if "date" in data and "weekday" in data:
                return f"当前时间是：{data['date']}，{data['weekday']}"
            else:
                return "无法获取当前时间，API返回格式异常"
        except Exception as e:
            return f"查询时间时发生错误：{e}"


class BasicCalculatorTool(ToolBase):
    """基本计算器工具"""
    
    def __init__(self):
        self.calc = Calculator()
    
    @property
    def name_for_human(self) -> str:
        return '基本计算器'
    
    @property
    def name_for_model(self) -> str:
        return 'basic_calculator'
    
    @property
    def description_for_model(self) -> str:
        return '用于基本数学运算的计算器,支持加减乘除、幂运算、平方根、绝对值、指数、对数等。可以计算数学表达式'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'expression',
                'description': '要计算的数学表达式,例如"2+3*4"、"sqrt(16)"、"abs(-5)"、"2**3"等。支持运算符:+,-,*,/,**。支持函数:sqrt(平方根),abs(绝对值),exp(指数),log(对数)。支持常数:pi,e',
                'required': True,
                'schema': {'type': 'string'},
            }
        ]
    
    def execute(self, expression: str) -> str:
        logger.info(f"Basic calculator: {expression}")
        
        try:
            result = self.calc.evaluate(expression)
            logger.success(f"Calculation result: {result}")
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return f"计算失败: {str(e)}"


class TrigCalculatorTool(ToolBase):
    """三角函数计算器工具"""
    
    def __init__(self):
        self.calc = Calculator()
    
    @property
    def name_for_human(self) -> str:
        return '三角函数计算器'
    
    @property
    def name_for_model(self) -> str:
        return 'trig_calculator'
    
    @property
    def description_for_model(self) -> str:
        return '用于计算三角函数和反三角函数。支持sin(正弦)、cos(余弦)、tan(正切)、asin(反正弦)、acos(反余弦)、atan(反正切)'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
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
    
    def execute(self, function: str, x: float, degree: bool = False) -> str:
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


class MatrixCalculatorTool(ToolBase):
    """矩阵计算器工具"""
    
    def __init__(self):
        self.calc = Calculator()
    
    @property
    def name_for_human(self) -> str:
        return '矩阵计算器'
    
    @property
    def name_for_model(self) -> str:
        return 'matrix_calculator'
    
    @property
    def description_for_model(self) -> str:
        return '用于矩阵运算,支持矩阵加法、减法、乘法、数乘、求行列式、求逆矩阵'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
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
    
    def execute(self, operation: str, matrix_a: List[List[float]], 
                matrix_b: List[List[float]] = None, scalar: float = None) -> str:
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


class IntegralCalculatorTool(ToolBase):
    """定积分计算器工具"""
    
    def __init__(self):
        self.calc = Calculator()
    
    @property
    def name_for_human(self) -> str:
        return '定积分计算器'
    
    @property
    def name_for_model(self) -> str:
        return 'integral_calculator'
    
    @property
    def description_for_model(self) -> str:
        return '用于计算定积分。可以计算各种函数在指定区间上的定积分'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
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
    
    def execute(self, func_str: str, a: float, b: float) -> str:
        try:
            result = self.calc.integrate_function(func_str, a, b)
            return result['description']
        except Exception as e:
            return f"积分计算失败: {str(e)}"


class KnowledgeBaseQueryTool(ToolBase):
    """知识库问答工具"""
    
    def __init__(self, rag_db_path: Optional[str] = None):
        self.rag_engine = None
        if rag_db_path and os.path.exists(rag_db_path):
            try:
                self.rag_engine = RAGEngine(rag_db_path)
                logger.info(f"RAG engine initialized with database: {rag_db_path}")
            except Exception as e:
                logger.error(f"RAG engine initialization failed: {e}")
    
    @property
    def name_for_human(self) -> str:
        return '知识库问答'
    
    @property
    def name_for_model(self) -> str:
        return 'knowledge_base_query'
    
    @property
    def description_for_model(self) -> str:
        return '一个基于向量检索的知识库问答工具,可以从预先构建的文档库中检索相关信息来回答问题。当用户询问特定领域知识、文档内容、或需要基于已有资料回答时使用此工具。如果使用网络查询失败,可以尝试使用此工具从知识库中获取答案。'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
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
    
    def execute(self, question: str, top_k: int = 3) -> str:
        logger.info(f"Knowledge base query: {question[:100]}...")
        
        if self.rag_engine is None:
            logger.warning("RAG engine not initialized")
            return "知识库未加载,无法回答问题。请先构建并加载向量数据库。"
        
        try:
            result = self.rag_engine.query(question, top_k=top_k)
            context = result['context']
            logger.success(f"Retrieved {len(result['search_results'])} results from knowledge base")
            
            return f"根据知识库检索到以下相关信息:\n\n{context}"
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            return f"知识库查询失败: {str(e)}"


class GetToolsInfoTool(ToolBase):
    """获取所有可用工具信息的工具"""
    
    def __init__(self, tools_manager_ref=None):
        self._manager_ref = tools_manager_ref
    
    @property
    def name_for_human(self) -> str:
        return '工具信息查询'
    
    @property
    def name_for_model(self) -> str:
        return 'get_tools_info'
    
    @property
    def description_for_model(self) -> str:
        return '获取当前所有可用工具的详细信息，包括工具名称、描述和参数说明。当你需要了解有哪些工具可用，或者不确定某个工具的使用方法时，可以调用此工具。'
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return []
    
    def execute(self) -> str:
        if self._manager_ref is None:
            return "错误: 工具管理器引用未设置"
        
        tools_info = self._manager_ref.get_all_tools_info()
        
        # 格式化工具信息
        result = f"当前共有 {len(tools_info)} 个可用工具:\n\n"
        
        for i, tool in enumerate(tools_info, 1):
            result += f"{i}. {tool['name_for_model']} ({tool['name_for_human']})\n"
            result += f"   描述: {tool['description_for_model']}\n"
            
            if tool['parameters']:
                result += f"   参数:\n"
                for param in tool['parameters']:
                    required = "必需" if param.get('required', False) else "可选"
                    param_type = param.get('schema', {}).get('type', 'unknown')
                    result += f"     - {param['name']} ({param_type}, {required}): {param['description']}\n"
            else:
                result += f"   参数: 无\n"
            result += "\n"
        
        return result.strip()


class ToolsManager:
    """工具管理器，提供统一的工具调用接口"""
    
    def __init__(self, rag_db_path: Optional[str] = None):
        # 初始化所有工具
        self._tools: Dict[str, ToolBase] = {}
        
        # 注册所有工具
        self._register_tool(GoogleSearchTool())
        self._register_tool(TavilySearchTool())
        self._register_tool(WeatherQueryTool())
        self._register_tool(TimeQueryTool())
        self._register_tool(BasicCalculatorTool())
        self._register_tool(TrigCalculatorTool())
        self._register_tool(MatrixCalculatorTool())
        self._register_tool(IntegralCalculatorTool())
        self._register_tool(KnowledgeBaseQueryTool(rag_db_path))
        
        # 注册工具信息查询工具(需要引用自己)
        tools_info_tool = GetToolsInfoTool(tools_manager_ref=self)
        self._register_tool(tools_info_tool)
        
        logger.info(f"ToolsManager initialized with {len(self._tools)} tools")
    
    def _register_tool(self, tool: ToolBase):
        """注册工具"""
        self._tools[tool.name_for_model] = tool
        logger.debug(f"Registered tool: {tool.name_for_model}")
    
    def get_all_tools_info(self) -> List[Dict[str, Any]]:
        """获取所有工具的配置信息"""
        return [tool.get_config() for tool in self._tools.values()]
    
    def get_tool_names(self) -> List[str]:
        """获取所有工具的模型调用名称"""
        return list(self._tools.keys())
    
    def call_tool(self, tool_name: str, **kwargs) -> str:
        """统一的工具调用接口
        
        Args:
            tool_name: 工具名称(name_for_model)
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        if tool_name not in self._tools:
            logger.error(f"Tool not found: {tool_name}")
            return f"错误: 未找到工具 '{tool_name}'"
        
        tool = self._tools[tool_name]
        timeout = tool.get_timeout()
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"工具执行超时 (限制时间: {timeout}秒)")
            
            # 设置信号处理器
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                result = tool.execute(**kwargs)
                signal.alarm(0)  # 取消闹钟
                logger.info(f"Tool '{tool_name}' executed successfully (timeout: {timeout}s)")
                return result
            finally:
                signal.alarm(0)  # 确保取消闹钟
                signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器
                
        except TimeoutError as e:
            logger.error(f"Tool '{tool_name}' execution timeout: {e}")
            return f"工具 '{tool_name}' 执行超时: {str(e)}"
        except TypeError as e:
            logger.error(f"Tool '{tool_name}' parameter error: {e}")
            return f"工具 '{tool_name}' 参数错误: {str(e)}"
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")
            return f"工具 '{tool_name}' 执行失败: {str(e)}"
    
    # 保留向后兼容的属性
    @property
    def toolConfig(self) -> List[Dict[str, Any]]:
        """向后兼容: 返回工具配置列表"""
        return self.get_all_tools_info()