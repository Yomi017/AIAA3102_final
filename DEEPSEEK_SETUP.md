# DeepSeek API 设置指南

## 为什么使用 DeepSeek？

- **成本更低**: 相比 Google Gemini，DeepSeek API 价格更实惠
- **兼容性好**: 使用 OpenAI SDK，接口简单易用
- **性能优秀**: DeepSeek-V3.2 提供强大的推理能力

## 快速设置

### 1. 获取 API Key

1. 访问 DeepSeek 平台: https://platform.deepseek.com/
2. 注册/登录账号
3. 进入 API Keys 页面创建新的 key
4. 复制你的 API key

### 2. 安装依赖

```bash
pip install openai
```

### 3. 设置环境变量

**Linux/Mac**:
```bash
export DEEPSEEK_API_KEY='your-deepseek-api-key'
```

**Windows (PowerShell)**:
```powershell
$env:DEEPSEEK_API_KEY='your-deepseek-api-key'
```

**永久设置** (添加到 `~/.bashrc` 或 `~/.zshrc`):
```bash
echo 'export DEEPSEEK_API_KEY="your-deepseek-api-key"' >> ~/.bashrc
source ~/.bashrc
```

### 4. 验证设置

```bash
echo $DEEPSEEK_API_KEY
```

### 5. 测试 API

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-api-key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```

## API 参数

| 参数 | 值 |
|------|-----|
| base_url | `https://api.deepseek.com` (或 `https://api.deepseek.com/v1`) |
| api_key | 你的 API key |
| model | `deepseek-chat` (通用模式) 或 `deepseek-reasoner` (推理模式) |

## 模型说明

- **deepseek-chat**: DeepSeek-V3.2 非思考模式，适合快速响应
- **deepseek-reasoner**: DeepSeek-V3.2 思考模式，适合复杂推理

## 在本项目中使用

### 运行通用能力测试

```bash
# 设置 API key
export DEEPSEEK_API_KEY='your-api-key'

# 运行测试
python test_general_capability.py --type rag --max_cases 3
```

测试脚本会自动使用 DeepSeek API 对 Agent 的回答进行评分（0-1 分）。

## 常见问题

### Q: API Key 在哪里获取？
A: https://platform.deepseek.com/ → API Keys

### Q: 如何查看 API 用量和费用？
A: 登录 DeepSeek 平台 → Usage Dashboard

### Q: 出现 "API key not set" 错误？
A: 确保已设置环境变量: `export DEEPSEEK_API_KEY='your-key'`

### Q: 出现 "Connection Error"？
A: 检查网络连接，确保可以访问 https://api.deepseek.com

### Q: 如何切换回 Gemini？
A: 修改代码中的 API 调用部分，或联系开发者

## 价格参考

DeepSeek API 定价通常比其他主流 LLM API 更实惠。具体价格请查看官方文档：
https://platform.deepseek.com/api-docs/pricing/

## 更多信息

- 官方文档: https://platform.deepseek.com/api-docs/
- 快速开始: https://platform.deepseek.com/api-docs/quick_start/
