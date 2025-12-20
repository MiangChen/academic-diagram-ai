"""
AcademicArchitect - 将论文内容转化为 Visual Schema
"""

import requests

from .config import load_config
from .constants import LAYOUT_TYPES
from .prompts import ARCHITECT_PROMPT


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
                "layout_hint": (list(LAYOUT_TYPES.keys()), {"default": "Linear Pipeline"}),
                "zone_count": ("INT", {"default": 3, "min": 2, "max": 5}),
            },
            "optional": {
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("visual_schema", "full_prompt")
    FUNCTION = "generate_schema"
    CATEGORY = "DMXAPI/Academic"

    def generate_schema(self, paper_content, layout_hint, zone_count, custom_instructions=""):
        config = load_config()
        api_key = config.get("api_key", "")
        url = config.get("api_url", "https://vip.dmxapi.com/v1/chat/completions")
        
        if not api_key:
            return ("错误: 未配置 API Key", "")
        
        full_prompt = ARCHITECT_PROMPT + paper_content
        if custom_instructions:
            full_prompt += f"\n\n# Additional Instructions\n{custom_instructions}"
        full_prompt += f"\n\n# Hints\n- 建议使用 {layout_hint} 布局\n- 建议划分 {zone_count} 个区域"
        
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
                return (schema, full_prompt)
            else:
                error = f"API 错误: {response.status_code}"
                print(f"[Architect] {error}")
                return (error, full_prompt)
                
        except Exception as e:
            error = f"错误: {str(e)}"
            print(f"[Architect] {error}")
            return (error, full_prompt)
