import numpy as np
from scipy import integrate
from typing import Union, List
import math

class Calculator:
    
    # ==================== 基本四则运算 ====================
    @staticmethod
    def add(a: Union[float, int], b: Union[float, int]) -> float:
        """加法"""
        return a + b
    
    @staticmethod
    def subtract(a: Union[float, int], b: Union[float, int]) -> float:
        """减法"""
        return a - b
    
    @staticmethod
    def multiply(a: Union[float, int], b: Union[float, int]) -> float:
        """乘法"""
        return a * b
    
    @staticmethod
    def divide(a: Union[float, int], b: Union[float, int]) -> float:
        """除法"""
        if b == 0:
            raise ValueError("除数不能为0")
        return a / b
    
    @staticmethod
    def power(a: Union[float, int], b: Union[float, int]) -> float:
        """幂运算"""
        return a ** b
    
    # ==================== 绝对值 ====================
    @staticmethod
    def abs_value(x: Union[float, int]) -> float:
        """绝对值"""
        return abs(x)
    
    # ==================== 三角函数 ====================
    @staticmethod
    def sin(x: Union[float, int], degree: bool = False) -> float:
        """正弦函数
        Args:
            x: 角度值
            degree: 是否使用角度制(默认False,使用弧度制)
        """
        if degree:
            x = math.radians(x)
        return math.sin(x)
    
    @staticmethod
    def cos(x: Union[float, int], degree: bool = False) -> float:
        """余弦函数
        Args:
            x: 角度值
            degree: 是否使用角度制(默认False,使用弧度制)
        """
        if degree:
            x = math.radians(x)
        return math.cos(x)
    
    @staticmethod
    def tan(x: Union[float, int], degree: bool = False) -> float:
        """正切函数
        Args:
            x: 角度值
            degree: 是否使用角度制(默认False,使用弧度制)
        """
        if degree:
            x = math.radians(x)
        return math.tan(x)
    
    # ==================== 反三角函数 ====================
    @staticmethod
    def asin(x: Union[float, int], degree: bool = False) -> float:
        """反正弦函数
        Args:
            x: 输入值,范围[-1, 1]
            degree: 是否返回角度制(默认False,返回弧度制)
        """
        if not -1 <= x <= 1:
            raise ValueError("asin的输入值必须在[-1, 1]范围内")
        result = math.asin(x)
        return math.degrees(result) if degree else result
    
    @staticmethod
    def acos(x: Union[float, int], degree: bool = False) -> float:
        """反余弦函数
        Args:
            x: 输入值,范围[-1, 1]
            degree: 是否返回角度制(默认False,返回弧度制)
        """
        if not -1 <= x <= 1:
            raise ValueError("acos的输入值必须在[-1, 1]范围内")
        result = math.acos(x)
        return math.degrees(result) if degree else result
    
    @staticmethod
    def atan(x: Union[float, int], degree: bool = False) -> float:
        """反正切函数
        Args:
            x: 输入值
            degree: 是否返回角度制(默认False,返回弧度制)
        """
        result = math.atan(x)
        return math.degrees(result) if degree else result
    
    # ==================== 矩阵运算 ====================
    @staticmethod
    def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """矩阵加法"""
        A_np = np.array(A)
        B_np = np.array(B)
        if A_np.shape != B_np.shape:
            raise ValueError(f"矩阵维度不匹配: {A_np.shape} vs {B_np.shape}")
        result = A_np + B_np
        return result.tolist()
    
    @staticmethod
    def matrix_subtract(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """矩阵减法"""
        A_np = np.array(A)
        B_np = np.array(B)
        if A_np.shape != B_np.shape:
            raise ValueError(f"矩阵维度不匹配: {A_np.shape} vs {B_np.shape}")
        result = A_np - B_np
        return result.tolist()
    
    @staticmethod
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """矩阵乘法"""
        A_np = np.array(A)
        B_np = np.array(B)
        if A_np.shape[1] != B_np.shape[0]:
            raise ValueError(f"矩阵维度不匹配,无法相乘: {A_np.shape} × {B_np.shape}")
        result = np.matmul(A_np, B_np)
        return result.tolist()
    
    @staticmethod
    def matrix_scalar_multiply(A: List[List[float]], scalar: Union[float, int]) -> List[List[float]]:
        """矩阵数乘"""
        A_np = np.array(A)
        result = A_np * scalar
        return result.tolist()
    
    @staticmethod
    def matrix_determinant(A: List[List[float]]) -> float:
        """计算矩阵行列式"""
        A_np = np.array(A)
        if A_np.shape[0] != A_np.shape[1]:
            raise ValueError("只有方阵才能计算行列式")
        return float(np.linalg.det(A_np))
    
    @staticmethod
    def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
        """计算矩阵的逆"""
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
        """计算定积分
        Args:
            func_str: 函数表达式字符串,例如 "x**2" 或 "np.sin(x)"
            a: 积分下限
            b: 积分上限
        Returns:
            dict: 包含积分结果和误差估计
        """
        # 创建一个安全的命名空间,只包含必要的函数
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
    
    # ==================== 通用计算接口 ====================
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


# 测试代码
if __name__ == "__main__":
    calc = Calculator()
    
    print("=== 基本四则运算 ===")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"5 - 3 = {calc.subtract(5, 3)}")
    print(f"5 × 3 = {calc.multiply(5, 3)}")
    print(f"5 ÷ 3 = {calc.divide(5, 3):.4f}")
    print(f"2^3 = {calc.power(2, 3)}")
    
    print("\n=== 绝对值 ===")
    print(f"|-5| = {calc.abs_value(-5)}")
    
    print("\n=== 三角函数 ===")
    print(f"sin(30°) = {calc.sin(30, degree=True):.4f}")
    print(f"cos(π/3) = {calc.cos(math.pi/3):.4f}")
    print(f"tan(45°) = {calc.tan(45, degree=True):.4f}")
    
    print("\n=== 反三角函数 ===")
    print(f"arcsin(0.5) = {calc.asin(0.5):.4f} 弧度")
    print(f"arccos(0.5) = {calc.acos(0.5, degree=True):.4f}°")
    print(f"arctan(1) = {calc.atan(1, degree=True):.4f}°")
    
    print("\n=== 矩阵运算 ===")
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"A + B = {calc.matrix_add(A, B)}")
    print(f"A - B = {calc.matrix_subtract(A, B)}")
    print(f"A × B = {calc.matrix_multiply(A, B)}")
    print(f"2A = {calc.matrix_scalar_multiply(A, 2)}")
    print(f"det(A) = {calc.matrix_determinant(A)}")
    
    print("\n=== 定积分 ===")
    # ∫x²dx from 0 to 1
    result = calc.integrate_function("x**2", 0, 1)
    print(f"{result['description']}")
    print(f"误差估计: {result['error']:.2e}")
    
    # ∫sin(x)dx from 0 to π
    result = calc.integrate_function("np.sin(x)", 0, np.pi)
    print(f"{result['description']}")
    print(f"误差估计: {result['error']:.2e}")
    
    print("\n=== 表达式计算 ===")
    print(f"2 + 3 * 4 = {calc.evaluate('2 + 3 * 4')}")
    print(f"sqrt(16) + abs(-5) = {calc.evaluate('sqrt(16) + abs(-5)')}")
