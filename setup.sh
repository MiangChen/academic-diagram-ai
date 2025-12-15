#!/bin/bash
# ComfyUI 环境安装脚本
# 用法: bash setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$SCRIPT_DIR/ComfyUI"
CONFIG_FILE="$SCRIPT_DIR/config_llm.json"
CONFIG_EXAMPLE="$SCRIPT_DIR/config_llm_example.json"
SETUP_MARKER="$SCRIPT_DIR/.setup_done"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== ComfyUI 环境安装 ===${NC}"

# 安装 ComfyUI 依赖
echo -e "${YELLOW}[1/2] 安装 ComfyUI requirements...${NC}"
if ! pip install -r "$COMFYUI_DIR/requirements.txt"; then
    echo -e "${RED}[错误] pip install 失败${NC}"
    echo -e "${YELLOW}可能的原因:${NC}"
    echo -e "${YELLOW}  - 未安装 Python 或 pip${NC}"
    echo -e "${YELLOW}  - 网络问题，请检查网络连接或使用镜像源${NC}"
    echo -e "${YELLOW}  - 权限问题，尝试使用 pip install --user${NC}"
    exit 1
fi

# 检查配置文件
echo -e "${YELLOW}[2/2] 检查配置文件...${NC}"
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
echo -e "${YELLOW}下一步: 编辑 config_llm.json 填入 API Key，然后运行 bash start.sh${NC}"
