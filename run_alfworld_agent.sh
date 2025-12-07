#!/bin/bash
################################################################################
# ALFworld Agent 测试脚本 - 使用完整 Agent 框架
# 
# 使用方法:
#   ./run_alfworld_agent.sh              # 交互式询问测试数量
#   ./run_alfworld_agent.sh 10           # 测试 10 个游戏
################################################################################

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════"
    echo "   🎮 ALFworld Agent 测试 (完整框架: ReAct + 工具)"
    echo "═══════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# 检查环境
check_environment() {
    print_info "检查运行环境..."
    
    # 检查是否在项目根目录
    if [ ! -f "benchmarkTest/run_agent_test.py" ]; then
        print_error "请在项目根目录 (/data/home/sim6g/AIAA3102_final/) 运行此脚本"
        exit 1
    fi
    
    # 检查 Python 环境
    if ! command -v python &> /dev/null; then
        print_error "未找到 Python 环境"
        exit 1
    fi
    
    # 检查 ALFworld 是否安装
    python -c "import alfworld" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "ALFworld 未安装"
        echo ""
        echo "请运行以下命令安装 ALFworld:"
        echo "  pip install alfworld[full]"
        echo "  alfworld-download  # 下载游戏数据"
        exit 1
    fi
    
    # 检查模型文件
    if [ ! -f "Qwen3-8B/config.json" ]; then
        print_warning "未找到模型文件 Qwen3-8B/，请确保模型路径正确"
    fi
    
    # 检查配置文件
    if [ ! -f "configs/base_config.yaml" ]; then
        print_error "未找到 ALFworld 配置文件: configs/base_config.yaml"
        exit 1
    fi
    
    print_success "环境检查通过"
}

# 主函数
main() {
    print_banner
    check_environment
    
    # 解析参数
    if [ $# -eq 0 ]; then
        NUM_GAMES=""
        print_info "未指定游戏数量，将在运行时交互式询问"
    else
        NUM_GAMES="--num_games $1"
        print_info "测试游戏数量: $1"
    fi
    
    # 配置参数
    GPU_IDS="4,5,6,7"  # 默认使用 GPU 4-7
    CONFIG_PATH="configs/base_config.yaml"
    
    # 显示配置
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}测试配置:${NC}"
    echo "  • 环境: ALFworld 文本游戏"
    echo "  • 配置文件: $CONFIG_PATH"
    echo "  • 使用 GPU: $GPU_IDS"
    echo "  • 测试类型: Agent (完整 ReAct 框架)"
    echo "  • 特性: ✅ ReAct 推理 | ✅ 工具调用 | ✅ 多轮优化"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # 询问确认
    read -p "是否开始测试? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "测试已取消"
        exit 0
    fi
    
    # 运行测试
    print_info "启动 ALFworld Agent 测试..."
    echo ""
    
    python benchmarkTest/run_agent_test.py \
        --config $CONFIG_PATH \
        --gpu_ids $GPU_IDS \
        $NUM_GAMES
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo ""
        print_success "ALFworld Agent 测试完成！"
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}结果保存位置:${NC}"
        echo "  benchmark_results/alfworld_agent/session_XXXXXX/"
        echo ""
        echo -e "${GREEN}查看结果:${NC}"
        echo "  cd benchmark_results/alfworld_agent/"
        echo "  ls -lt  # 查看最新的 session 目录"
        echo "  cat session_*/statistics.txt  # 查看统计报告"
        echo "  cat session_*/summary.json    # 查看详细结果"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    else
        echo ""
        print_error "ALFworld Agent 测试失败，请检查日志"
        exit 1
    fi
}

# 执行主函数
main "$@"
