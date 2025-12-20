#!/bin/bash
# ComfyUI 启动脚本
# 用法: bash start.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$SCRIPT_DIR/ComfyUI"
CONFIG_FILE="$SCRIPT_DIR/config_llm.json"
SETUP_MARKER="$SCRIPT_DIR/.setup_done"
ENV_NAME="academic_diagram_311"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== ComfyUI 启动脚本 ===${NC}"

# 检查是否已运行过 setup
if [ ! -f "$SETUP_MARKER" ]; then
    echo -e "${RED}[错误] 尚未运行安装脚本${NC}"
    echo -e "${YELLOW}请先运行: bash setup.sh${NC}"
    exit 1
fi

# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo -e "${GREEN}[环境] 已激活 conda 环境: ${ENV_NAME}${NC}"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}[错误] 未找到配置文件: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}请先运行: bash setup.sh${NC}"
    exit 1
fi

# 进入 ComfyUI 目录并启动
cd "$COMFYUI_DIR"
echo -e "${GREEN}[启动] ComfyUI @ http://localhost:8188${NC}"
echo ""

python main.py --listen 0.0.0.0 --port 8188
