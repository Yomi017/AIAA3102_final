#!/bin/bash
# Agent 测试 - 完整框架（ReAct + 工具）
# 用法: ./run_agent_test.sh [ai|robomaster|web]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   🤖 Agent 测试 (ReAct + 工具)${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════${NC}"
}

check_environment() {
    if [ ! -f "test_rag_capability.py" ]; then
        print_error "请在 testsh/selfbenchmark/ 目录下运行"
        exit 1
    fi
    if ! command -v python &> /dev/null; then
        print_error "未找到 Python"
        exit 1
    fi
}

# 主函数
main() {
    print_banner
    check_environment
    
    # 解析参数
    if [ $# -eq 0 ]; then
        # 没有参数，测试所有数据集
        TEST_SETS="ai robomaster"
        print_info "未指定数据集，将测试所有数据集: ai, robomaster"
    else
        # 使用用户指定的数据集
        TEST_SETS="$@"
        print_info "测试数据集: $TEST_SETS"
    fi
    
    # 配置参数
    GPU_IDS="4,5,6,7"  # 默认使用 GPU 4-7
    MAX_CASES=""       # 不限制测试数量，如需限制可设置为 "--max_cases 5"
    
    # 检查是否设置了 DeepSeek API Key（用于 Web 测试评分）
    if [[ " $TEST_SETS " =~ " web " ]]; then
        if [ -z "$DEEPSEEK_API_KEY" ]; then
            print_warning "未设置 DEEPSEEK_API_KEY 环境变量"
            print_warning "Web 搜索测试需要 DeepSeek API 进行评分"
            print_warning "或者在 config.yaml 中配置 DEEPSEEK_API 字段"
            echo ""
        else
            print_success "检测到 DEEPSEEK_API_KEY 环境变量"
        fi
    fi
    
    # 显示配置
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}测试配置:${NC}"
    echo "  • 测试数据集: $TEST_SETS"
    echo "  • 使用 GPU: $GPU_IDS"
    echo "  • 测试类型: Agent (完整框架)"
    echo "  • 评分方式: AI/RoboMaster=F1 Score, Web=DeepSeek API"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # 询问确认
    read -p "是否开始测试? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "测试已取消"
        exit 0
    fi
    
    # 运行测试
    print_info "开始 Agent 测试..."
    echo ""
    
    python test_rag_capability.py \
        --test_sets $TEST_SETS \
        --gpu_ids $GPU_IDS \
        $MAX_CASES
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Agent 测试完成！"
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}结果保存位置:${NC}"
        echo "  ../../benchmark_results/agent/ai/session_XXXXXX/"
        echo "  ../../benchmark_results/agent/robomaster/session_XXXXXX/"
        echo "  ../../benchmark_results/agent/web/session_XXXXXX/"
        echo ""
        echo -e "${GREEN}查看结果:${NC}"
        echo "  cd ../../benchmark_results/agent/<dataset>/"
        echo "  cat session_*/summary.txt"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        echo ""
        print_error "Agent 测试失败，请检查日志"
        exit 1
    fi
}

# 执行主函数
main "$@"
