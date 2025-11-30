"""
测试 query_time API 的功能
这个文件用于测试 tool.py 中的 query_time 方法是否能正常工作
"""

import requests
import json
from typing import Dict, Any


def test_api_connectivity() -> bool:
    """
    测试API的连接性
    Returns:
        bool: 如果API可以连接返回True，否则返回False
    """
    url = "https://api.uuni.cn//api/time"
    try:
        response = requests.get(url, timeout=5)
        print(f"✓ API连接成功，状态码: {response.status_code}")
        return True
    except requests.exceptions.Timeout:
        print("✗ API连接超时")
        return False
    except requests.exceptions.ConnectionError:
        print("✗ 无法连接到API")
        return False
    except Exception as e:
        print(f"✗ 连接时发生错误: {e}")
        return False


def test_api_response_format() -> bool:
    """
    测试API返回的数据格式是否正确
    Returns:
        bool: 如果返回格式正确返回True，否则返回False
    """
    url = "https://api.uuni.cn//api/time"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        # 尝试解析JSON
        data = response.json()
        print(f"✓ API返回的JSON数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        
        # 检查必需的字段
        if "date" in data and "weekday" in data:
            print(f"✓ 数据格式正确，包含必需字段 'date' 和 'weekday'")
            return True
        else:
            print(f"✗ 数据格式异常，缺少必需字段")
            print(f"  期望字段: ['date', 'weekday']")
            print(f"  实际字段: {list(data.keys())}")
            return False
            
    except json.JSONDecodeError:
        print("✗ API返回的不是有效的JSON格式")
        return False
    except Exception as e:
        print(f"✗ 检查响应格式时发生错误: {e}")
        return False


def test_query_time_method() -> bool:
    """
    测试完整的 query_time 方法（模拟 tool.py 中的实现）
    Returns:
        bool: 如果方法正常工作返回True，否则返回False
    """
    url = "https://api.uuni.cn//api/time"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "date" in data and "weekday" in data:
            result = f"当前时间是：{data['date']}，{data['weekday']}"
            print(f"✓ query_time 方法测试成功")
            print(f"  返回结果: {result}")
            return True
        else:
            print("✗ query_time 方法测试失败：API返回格式异常")
            return False
            
    except Exception as e:
        print(f"✗ query_time 方法测试失败: {e}")
        return False


def test_api_data_validity() -> bool:
    """
    测试API返回的数据是否有效（检查数据类型和内容）
    Returns:
        bool: 如果数据有效返回True，否则返回False
    """
    url = "https://api.uuni.cn//api/time"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # 检查数据类型
        if not isinstance(data.get("date"), str):
            print(f"✗ 'date' 字段类型错误，期望 str，实际 {type(data.get('date'))}")
            return False
            
        if not isinstance(data.get("weekday"), str):
            print(f"✗ 'weekday' 字段类型错误，期望 str，实际 {type(data.get('weekday'))}")
            return False
        
        # 检查数据是否为空
        if not data.get("date") or not data.get("weekday"):
            print("✗ 'date' 或 'weekday' 字段为空")
            return False
        
        print(f"✓ 数据有效性检查通过")
        print(f"  日期: {data['date']}")
        print(f"  星期: {data['weekday']}")
        return True
        
    except Exception as e:
        print(f"✗ 数据有效性检查失败: {e}")
        return False


def run_all_tests():
    """
    运行所有测试
    """
    print("=" * 60)
    print("开始测试 query_time API")
    print("=" * 60)
    print()
    
    tests = [
        ("API连接性测试", test_api_connectivity),
        ("API响应格式测试", test_api_response_format),
        ("数据有效性测试", test_api_data_validity),
        ("query_time方法测试", test_query_time_method),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n【{test_name}】")
        print("-" * 60)
        result = test_func()
        results.append((test_name, result))
        print()
    
    # 打印测试总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {test_name}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！API工作正常。")
    else:
        print("⚠️  部分测试失败，请检查API状态。")
    
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
