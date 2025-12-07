#!/bin/bash
################################################################################
# Baseline 测试脚本 - 消融实验（无 Agent 框架）
# 
# 使用方法:
#   ./run_baseline_test.sh                    # 测试所有数据集
#   ./run_baseline_test.sh ai                 # 只测试 AI 知识库
#   ./run_baseline_test.sh ai robomaster      # 测试 AI 和 RoboMaster
#   ./run_baseline_test.sh web                # 只测试 Web 搜索
################################################################################

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印横幅
print_banner() {
    echo -e "${MAGENTA}"
    echo "═══════════════════════════════════════════════════════════════════"
    echo "   🧪 Baseline 测试脚本 (消融实验: 无 Agent 框架)"
    echo "═══════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# 检查环境
check_environment() {
    print_info "检查运行环境..."
    
    # 检查是否在正确的目录
    if [ ! -f "test_baseline_capability.py" ]; then
        print_error "请在 testsh/selfbenchmark/ 目录下运行此脚本"
        exit 1
    fi
    
    # 检查 Python 环境
    if ! command -v python &> /dev/null; then
        print_error "未找到 Python 环境"
        exit 1
    fi
    
    # 检查必要文件
    if [ ! -f "../../Qwen3-8B/config.json" ]; then
        print_warning "未找到模型文件 Qwen3-8B/，请确保模型路径正确"
    fi
    
    print_success "环境检查通过"
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
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}测试配置:${NC}"
    echo "  • 测试数据集: $TEST_SETS"
    echo "  • 使用 GPU: $GPU_IDS"
    echo "  • 测试类型: Baseline (消融实验)"
    echo "  • 特性: ❌ 无 ReAct | ❌ 无工具 | ❌ 无多轮优化"
    echo "  • 评分方式: AI/RoboMaster=F1 Score, Web=DeepSeek API"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # 询问确认
    read -p "是否开始测试? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "测试已取消"
        exit 0
    fi
    
    # 运行测试
    print_info "开始 Baseline 测试..."
    echo ""
    
    python test_baseline_capability.py \
        --test_sets $TEST_SETS \
        --gpu_ids $GPU_IDS \
        $MAX_CASES
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Baseline 测试完成！"
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}结果保存位置:${NC}"
        echo "  ../../benchmark_results/baseline/ai/session_XXXXXX/"
        echo "  ../../benchmark_results/baseline/robomaster/session_XXXXXX/"
        echo "  ../../benchmark_results/baseline/web/session_XXXXXX/"
        echo ""
        echo -e "${GREEN}查看结果:${NC}"
        echo "  cd ../../benchmark_results/baseline/<dataset>/"
        echo "  cat session_*/summary.txt"
        echo ""
        echo -e "${YELLOW}对比 Agent 和 Baseline:${NC}"
        echo "  python compare_results.py \\"
        echo "    ../../benchmark_results/agent/<dataset>/session_XXXXXX \\"
        echo "    ../../benchmark_results/baseline/<dataset>/session_XXXXXX"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        echo ""
        print_error "Baseline 测试失败，请检查日志"
        exit 1
    fi
}

# 执行主函数
main "$@"
