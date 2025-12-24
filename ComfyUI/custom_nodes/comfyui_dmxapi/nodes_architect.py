"""
AcademicArchitect - 将论文内容转化为 Visual Schema
"""

import requests

from .config import load_config
from .constants import LAYOUT_TYPES, COLOR_SCHEMES
from .prompts import get_architect_prompt


class AcademicArchitect:
    """
    Step 1: The Architect - 将论文内容转化为 Visual Schema
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paper_content": ("STRING", {
                    "multiline": True,
                    "default": "在此粘贴论文摘要或方法章节内容..."
                }),
                "layout_hint": (list(LAYOUT_TYPES.keys()), {"default": "None / 无"}),
                "zone_count / 分栏数量": ("INT", {"default": 3, "min": 0, "max": 10}),
                "color_scheme": (list(COLOR_SCHEMES.keys()), {"default": "无"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    # 关键：告诉 ComfyUI 这个节点的输出不是确定性的
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 返回随机值，强制每次都重新执行
        import random
        return random.random()
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("visual_schema", "color_scheme_text", "full_prompt")
    FUNCTION = "generate_schema"
    CATEGORY = "DMXAPI/Academic"

    def generate_schema(self, paper_content, layout_hint, color_scheme="无", seed=0, custom_instructions="", **kwargs):
        # 支持双语参数名
        zone_count = kwargs.get("zone_count / 分栏数量", 0)
        
        config = load_config()
        api_key = config.get("api_key", "")
        url = config.get("api_url", "https://vip.dmxapi.com/v1/chat/completions")
        
        if not api_key:
            return ("错误: 未配置 API Key", "", "")
        
        # 获取颜色方案文本
        color_scheme_text = COLOR_SCHEMES.get(color_scheme, "")
        use_preset_color = color_scheme != "无" and color_scheme_text
        
        # 根据是否使用预设颜色，生成不同的 prompt
        architect_prompt = get_architect_prompt(use_preset_color=use_preset_color)
        
        full_prompt = architect_prompt + paper_content
        if custom_instructions:
            full_prompt += f"\n\n# Additional Instructions\n{custom_instructions}"
        
        # 只有在非 "None / 无" 时才添加布局和分栏提示
        hints = []
        if layout_hint != "None / 无":
            hints.append(f"建议使用 {layout_hint} 布局")
        if zone_count > 0:
            hints.append(f"建议划分 {zone_count} 个区域")
        
        if hints:
            full_prompt += f"\n\n# Hints\n- " + "\n- ".join(hints)
        
        # 如果选择了预设颜色，强制指定
        if use_preset_color:
            full_prompt += f"\n\n# 颜色方案（必须严格遵守，不要自己发挥）\n{color_scheme_text}"
            print(f"[Architect] 使用预设颜色方案: {color_scheme}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        text_model = config.get("text_model", "gemini-3-pro-preview")
        
        payload = {
            "model": text_model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.7,
        }
        
        print(f"[Architect] 使用模型: {text_model}")
        
        try:
            print("[Architect] 正在生成 Visual Schema...")
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                schema = result["choices"][0]["message"]["content"]
                print("[Architect] Schema 生成成功")
                return (schema, color_scheme_text, full_prompt)
            else:
                error = f"API 错误: {response.status_code}"
                print(f"[Architect] {error}")
                return (error, color_scheme_text, full_prompt)
                
        except Exception as e:
            error = f"错误: {str(e)}"
            print(f"[Architect] {error}")
            return (error, color_scheme_text, full_prompt)
