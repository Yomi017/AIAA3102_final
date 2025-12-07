#!/bin/bash
################################################################################
# 主测试脚本 - 选择要运行的测试类型
################################################################################

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 打印主菜单
print_menu() {
    clear
    echo -e "${BOLD}${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║        🤖 AIAA3102 Agent 测试系统                            ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${CYAN}请选择要运行的测试:${NC}"
    echo ""
    echo -e "  ${GREEN}[1]${NC} RAG/Web Agent 测试       (知识检索 + 网络搜索)"
    echo -e "  ${MAGENTA}[2]${NC} RAG/Web Baseline 测试    (消融实验)"
    echo ""
    echo -e "  ${GREEN}[3]${NC} ALFworld Agent 测试      (交互式游戏)"
    echo -e "  ${MAGENTA}[4]${NC} ALFworld Baseline 测试   (消融实验)"
    echo ""
    echo -e "  ${YELLOW}[5]${NC} 运行所有 Agent 测试"
    echo -e "  ${YELLOW}[6]${NC} 运行所有 Baseline 测试"
    echo ""
    echo -e "  ${RED}[0]${NC} 退出"
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# 等待用户按键
press_any_key() {
    echo ""
    read -n 1 -s -r -p "按任意键继续..."
    echo ""
}

# RAG/Web Agent 测试
run_rag_agent() {
    echo -e "\n${GREEN}>>> 启动 RAG/Web Agent 测试${NC}\n"
    cd testsh/selfbenchmark
    ./run_agent_test.sh "$@"
    cd ../..
    press_any_key
}

# RAG/Web Baseline 测试
run_rag_baseline() {
    echo -e "\n${MAGENTA}>>> 启动 RAG/Web Baseline 测试${NC}\n"
    cd testsh/selfbenchmark
    ./run_baseline_test.sh "$@"
    cd ../..
    press_any_key
}

# ALFworld Agent 测试
run_alfworld_agent() {
    echo -e "\n${GREEN}>>> 启动 ALFworld Agent 测试${NC}\n"
    ./run_alfworld_agent.sh "$@"
    press_any_key
}

# ALFworld Baseline 测试
run_alfworld_baseline() {
    echo -e "\n${MAGENTA}>>> 启动 ALFworld Baseline 测试${NC}\n"
    ./run_alfworld_baseline.sh "$@"
    press_any_key
}

# 运行所有 Agent 测试
run_all_agent() {
    echo -e "\n${YELLOW}>>> 运行所有 Agent 测试${NC}\n"
    
    echo -e "${GREEN}━━━ 1/2: RAG/Web Agent 测试 ━━━${NC}"
    cd testsh/selfbenchmark
    ./run_agent_test.sh
    cd ../..
    
    echo -e "\n${GREEN}━━━ 2/2: ALFworld Agent 测试 ━━━${NC}"
    ./run_alfworld_agent.sh
    
    echo -e "\n${GREEN}✅ 所有 Agent 测试完成！${NC}"
    press_any_key
}

# 运行所有 Baseline 测试
run_all_baseline() {
    echo -e "\n${YELLOW}>>> 运行所有 Baseline 测试${NC}\n"
    
    echo -e "${MAGENTA}━━━ 1/2: RAG/Web Baseline 测试 ━━━${NC}"
    cd testsh/selfbenchmark
    ./run_baseline_test.sh
    cd ../..
    
    echo -e "\n${MAGENTA}━━━ 2/2: ALFworld Baseline 测试 ━━━${NC}"
    ./run_alfworld_baseline.sh
    
    echo -e "\n${MAGENTA}✅ 所有 Baseline 测试完成！${NC}"
    press_any_key
}

# 主循环
main() {
    # 检查是否在项目根目录
    if [ ! -f "agent.py" ] || [ ! -f "base.py" ]; then
        echo -e "${RED}错误: 请在项目根目录运行此脚本${NC}"
        echo "当前目录: $(pwd)"
        echo "期望目录: /data/home/sim6g/AIAA3102_final/"
        exit 1
    fi
    
    while true; do
        print_menu
        read -p "请输入选项 [0-6]: " choice
        
        case $choice in
            1)
                run_rag_agent
                ;;
            2)
                run_rag_baseline
                ;;
            3)
                run_alfworld_agent
                ;;
            4)
                run_alfworld_baseline
                ;;
            5)
                run_all_agent
                ;;
            6)
                run_all_baseline
                ;;
            0)
                echo -e "\n${YELLOW}退出测试系统。再见！${NC}\n"
                exit 0
                ;;
            *)
                echo -e "\n${RED}无效选项，请重新选择${NC}"
                sleep 1
                ;;
        esac
    done
}

# 执行主函数
main
