#!/bin/bash
# ComfyUI + SAM2 环境安装脚本
# 用法: bash setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$SCRIPT_DIR/ComfyUI"
CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
SAM2_NODE_DIR="$CUSTOM_NODES_DIR/ComfyUI-segment-anything-2"
CONFIG_FILE="$SCRIPT_DIR/config_llm.json"
CONFIG_EXAMPLE="$SCRIPT_DIR/config_llm_example.json"
SETUP_MARKER="$SCRIPT_DIR/.setup_done"
ENV_NAME="academic_diagram_311"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 学术插图 AI 环境安装 ===${NC}"

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo -e "${RED}[错误] 未找到 conda，请先安装 Anaconda 或 Miniconda${NC}"
    exit 1
fi

# 创建 conda 环境
echo -e "${YELLOW}[1/5] 创建 conda 环境 ${ENV_NAME} (Python 3.11)...${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${GREEN}[Setup] 环境 ${ENV_NAME} 已存在，跳过创建${NC}"
else
    if ! conda create -n "$ENV_NAME" python=3.11 -y; then
        echo -e "${RED}[错误] conda 环境创建失败${NC}"
        exit 1
    fi
fi

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 安装 ComfyUI 依赖
echo -e "${YELLOW}[2/5] 安装 ComfyUI requirements...${NC}"
if ! pip install -r "$COMFYUI_DIR/requirements.txt"; then
    echo -e "${RED}[错误] ComfyUI pip install 失败${NC}"
    exit 1
fi

# 安装 ComfyUI-SAM2 节点 (Kijai 版本)
echo -e "${YELLOW}[3/5] 安装 ComfyUI-SAM2 节点...${NC}"
if [ ! -d "$SAM2_NODE_DIR" ]; then
    echo -e "${YELLOW}[Setup] 克隆 ComfyUI-segment-anything-2...${NC}"
    git clone https://github.com/kijai/ComfyUI-segment-anything-2.git "$SAM2_NODE_DIR"
else
    echo -e "${GREEN}[Setup] ComfyUI-SAM2 已存在，跳过克隆${NC}"
fi

# 安装 SAM2 节点依赖
if [ -f "$SAM2_NODE_DIR/requirements.txt" ]; then
    echo -e "${YELLOW}[Setup] 安装 SAM2 节点依赖...${NC}"
    pip install -r "$SAM2_NODE_DIR/requirements.txt"
fi

# 检测 GPU 并安装 PyTorch
echo -e "${YELLOW}[4/5] 检测 GPU 环境并配置 PyTorch...${NC}"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
    if [ -n "$CUDA_VERSION" ]; then
        echo -e "${GREEN}[Setup] 检测到 NVIDIA GPU，CUDA $CUDA_VERSION${NC}"
        echo -e "${GREEN}[Setup] 确保 PyTorch CUDA 版本已安装...${NC}"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${YELLOW}[Setup] 未检测到 CUDA，使用 CPU 版本 PyTorch${NC}"
    fi
else
    echo -e "${YELLOW}[Setup] 未检测到 NVIDIA GPU，使用 CPU 版本 PyTorch${NC}"
fi

# 检查配置文件
echo -e "${YELLOW}[5/5] 检查配置文件...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    if [ -f "$CONFIG_EXAMPLE" ]; then
        cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
        echo -e "${GREEN}[Setup] 已从示例创建配置文件: $CONFIG_FILE${NC}"
    else
        echo -e "${RED}[Setup] 未找到示例配置文件${NC}"
    fi
    echo -e "${YELLOW}[Setup] 请编辑 $CONFIG_FILE 填入你的 API Key${NC}"
else
    echo -e "${GREEN}[Setup] 配置文件已存在: $CONFIG_FILE${NC}"
fi

# 创建安装标记
touch "$SETUP_MARKER"

echo ""
echo -e "${GREEN}=== 安装完成 ===${NC}"
echo -e "${YELLOW}使用方法:${NC}"
echo -e "${YELLOW}  1. 编辑 config_llm.json 填入 API Key${NC}"
echo -e "${YELLOW}  2. bash start.sh 启动 ComfyUI${NC}"
echo ""
echo -e "${GREEN}SAM2 使用:${NC}"
echo -e "${YELLOW}  在 ComfyUI 节点列表中搜索 SAM2 即可使用${NC}"
echo -e "${YELLOW}  首次使用需下载模型到 ComfyUI/models/sam2/${NC}"
