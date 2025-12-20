"""
AcademicEditor - 对生成的图像进行自然语言编辑
"""

import requests
import numpy as np
import torch
from PIL import Image
import io

from .config import load_config
from .utils import image_to_base64, parse_image_from_response


class AcademicEditor:
    """
    Step 3: The Editor - 对生成的图像进行自然语言编辑
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "edit_instruction": ("STRING", {
                    "multiline": True,
                    "default": "例如: Make all lines thinner, change the orange arrows to dark grey"
                }),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "edit"
    CATEGORY = "DMXAPI/Academic"

    def edit(self, image, edit_instruction, enabled=True):
        if not enabled:
            print("[Editor] 已停用，直接输出原图")
            return (image, "已停用 - 直接输出原图")
        
        config = load_config()
        api_key = config.get("api_key", "")
        url = config.get("api_url", "https://vip.dmxapi.com/v1/chat/completions")
        image_model = config.get("image_model", "gemini-2.5-flash-preview-05-20")
        
        if not api_key:
            return (image, "错误: 未配置 API Key")
        
        img_base64 = image_to_base64(image)
        
        edit_prompt = f"""You are editing an academic diagram. Please make the following modifications while preserving the overall structure and layout:

{edit_instruction}

Important:
- Keep the diagram style professional and suitable for academic papers
- Maintain clean lines and flat 2D design
- Do not add shadows or 3D effects
- Preserve all existing text labels unless specifically asked to change them"""
        
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            {"type": "text", "text": edit_prompt}
        ]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        payload = {
            "model": image_model,
            "messages": [{"role": "user", "content": content}],
        }
        
        info_text = "Editor 执行中..."
        
        try:
            print(f"[Editor] 使用模型: {image_model}")
            print(f"[Editor] 正在编辑: {edit_instruction[:50]}...")
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                resp_content = result["choices"][0]["message"]["content"]
                
                image_bytes = parse_image_from_response(resp_content)
                if image_bytes:
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    np_image = np.array(pil_image).astype(np.float32) / 255.0
                    info_text = f"编辑成功: {pil_image.size}"
                    print(f"[Editor] {info_text}")
                    return (torch.from_numpy(np_image).unsqueeze(0), info_text)
                else:
                    info_text = f"返回文本: {resp_content[:100]}"
            else:
                info_text = f"API 错误: {response.status_code}"
                
        except Exception as e:
            info_text = f"错误: {str(e)}"
        
        print(f"[Editor] {info_text}")
        return (image, info_text)
