# 学术插图 AI 绘制工作流

> 欢迎发起 issue，补充需要的绘图功能！

基于 ComfyUI 的学术插图生成工具，通过 LLM 理解需求并自动生成专业的学术风格图表。

## 环境配置

```bash
# 1. 安装依赖
bash setup.sh

# 2. 启动 ComfyUI
bash start.sh
```

首次运行前，请编辑 `config_llm.json` 填入你的 API Key 和 Base URL。

## 访问

http://localhost:8188

`Ctrl + O` 打开加载 workflow 的界面

![open.png](./asset/open_folder.png)

选择 `workflows/academic_diagram.json`

![image-20251215233812328](./asset/select_wf.png)

然后就可以看到该工作流：

![image-20251215233857972](./asset/workflow.png)

## 工作流概览

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Step 1         │    │  Step 2         │    │  Step 3         │
│  The Architect  │ -> │  The Renderer   │ -> │  The Editor     │
│  逻辑构建       │    │  视觉渲染       │    │  交互式微调     │
│  (Gemini/GPT)   │    │  (Nano-Banana)  │    │  (自然语言编辑) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 自定义节点

节点代码位于 `ComfyUI/custom_nodes/comfyui_dmxapi/`，包含三个核心节点：

| 节点 | 功能 |
|------|------|
| **AcademicArchitect** | 将论文内容（摘要/方法章节）转化为结构化的 Visual Schema |
| **AcademicRenderer** | 根据 Schema 调用图像模型生成学术风格插图，支持参考图输入 |
| **AcademicEditor** | 对生成的图像进行自然语言编辑微调 |

## 常见疑问

**Q: 我的 workflow 会被保存到哪里？**

A: 会被保存到 `/ComfyUI/user` 中



**Q: 为什么运行后是红色的结果**

A: 这是报错的意思.可能原因包括但不限于:

- API key不正确, 或者该API Key没有访问特定模型的权限
- API Base写错了, 可以把多种后缀的都试试, 比如 /v1,  /v1/chat/completions/的
- 其他报错, 可以在issue提

![](./asset/error.jpg)

