import numpy as np
from scipy import integrate
from typing import Union, List
import math

class Calculator:
    @staticmethod
    def add(a: Union[float, int], b: Union[float, int]) -> float:
        return a + b
    
    @staticmethod
    def subtract(a: Union[float, int], b: Union[float, int]) -> float:
        return a - b
    
    @staticmethod
    def multiply(a: Union[float, int], b: Union[float, int]) -> float:
        return a * b
    
    @staticmethod
    def divide(a: Union[float, int], b: Union[float, int]) -> float:
        if b == 0:
            raise ValueError("除数不能为0")
        return a / b
    
    @staticmethod
    def power(a: Union[float, int], b: Union[float, int]) -> float:
        return a ** b
    
    @staticmethod
    def abs_value(x: Union[float, int]) -> float:
        return abs(x)
        
    @staticmethod
    def sin(x: Union[float, int], degree: bool = False) -> float:
        if degree:
            x = math.radians(x)
        return math.sin(x)
    
    @staticmethod
    def cos(x: Union[float, int], degree: bool = False) -> float:
        if degree:
            x = math.radians(x)
        return math.cos(x)
    
    @staticmethod
    def tan(x: Union[float, int], degree: bool = False) -> float:
        if degree:
            x = math.radians(x)
        return math.tan(x)
    
    @staticmethod
    def asin(x: Union[float, int], degree: bool = False) -> float:
        if not -1 <= x <= 1:
            raise ValueError("asin的输入值必须在[-1, 1]范围内")
        result = math.asin(x)
        return math.degrees(result) if degree else result
    
    @staticmethod
    def acos(x: Union[float, int], degree: bool = False) -> float:
        if not -1 <= x <= 1:
            raise ValueError("acos的输入值必须在[-1, 1]范围内")
        result = math.acos(x)
        return math.degrees(result) if degree else result
    
    @staticmethod
    def atan(x: Union[float, int], degree: bool = False) -> float:
        result = math.atan(x)
        return math.degrees(result) if degree else result
    
    @staticmethod
    def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        A_np = np.array(A)
        B_np = np.array(B)
        if A_np.shape != B_np.shape:
            raise ValueError(f"矩阵维度不匹配: {A_np.shape} vs {B_np.shape}")
        result = A_np + B_np
        return result.tolist()
    
    @staticmethod
    def matrix_subtract(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        A_np = np.array(A)
        B_np = np.array(B)
        if A_np.shape != B_np.shape:
            raise ValueError(f"矩阵维度不匹配: {A_np.shape} vs {B_np.shape}")
        result = A_np - B_np
        return result.tolist()
    
    @staticmethod
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        A_np = np.array(A)
        B_np = np.array(B)
        if A_np.shape[1] != B_np.shape[0]:
            raise ValueError(f"矩阵维度不匹配,无法相乘: {A_np.shape} × {B_np.shape}")
        result = np.matmul(A_np, B_np)
        return result.tolist()
    
    @staticmethod
    def matrix_scalar_multiply(A: List[List[float]], scalar: Union[float, int]) -> List[List[float]]:
        A_np = np.array(A)
        result = A_np * scalar
        return result.tolist()
    
    @staticmethod
    def matrix_determinant(A: List[List[float]]) -> float:
        A_np = np.array(A)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只有方阵才能计算行列式")
        return float(np.linalg.det(A_np))
    
    @staticmethod
    def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
        A_np = np.array(A)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只有方阵才能计算逆矩阵")
        try:
            result = np.linalg.inv(A_np)
            return result.tolist()
        except np.linalg.LinAlgError:
            raise ValueError("矩阵不可逆")
    
    # ==================== 定积分 ====================
    @staticmethod
    def integrate_function(func_str: str, a: float, b: float) -> dict:
        safe_dict = {
            'np': np,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pi': np.pi,
            'e': np.e
        }
        
        try:
            # 定义函数
            func = lambda x: eval(func_str, {"__builtins__": {}}, {**safe_dict, 'x': x})
            
            # 计算定积分
            result, error = integrate.quad(func, a, b)
            
            return {
                'result': result,
                'error': error,
                'description': f"∫({func_str})dx from {a} to {b} = {result:.6f}"
            }
        except Exception as e:
            raise ValueError(f"积分计算失败: {str(e)}")

    @staticmethod
    def evaluate(expression: str) -> float:
        """计算数学表达式
        Args:
            expression: 数学表达式字符串,例如 "2 + 3 * 4"
        Returns:
            计算结果
        """
        safe_dict = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sqrt': math.sqrt,
            'abs': abs,
            'pi': math.pi,
            'e': math.e,
            'log': math.log,
            'exp': math.exp
        }
        
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return float(result)
        except Exception as e:
            raise ValueError(f"表达式计算失败: {str(e)}")

