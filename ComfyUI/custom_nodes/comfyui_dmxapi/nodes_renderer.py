"""
AcademicRenderer - 将 Visual Schema 渲染为图像
"""

import requests
import numpy as np
import torch
from PIL import Image
import io

from .config import load_config
from .utils import image_to_base64, parse_image_from_response, create_error_image


class AcademicRenderer:
    """
    Step 2: The Renderer - 将 Visual Schema 渲染为图像
    支持多张参考图片输入
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        image_inputs = {f"ref_image_{i}": ("IMAGE",) for i in range(1, 9)}
        
        return {
            "required": {
                "visual_schema": ("STRING", {
                    "multiline": True,
                    "default": "粘贴 Architect 生成的 Visual Schema 或直接输入绘图描述..."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            },
            "optional": {
                "color_scheme_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                }),
                **image_inputs
            }
        }
    
    # 强制每次重新执行（不使用缓存）
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import random
        return random.random()
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "render"
    CATEGORY = "DMXAPI/Academic"

    def render(self, visual_schema, seed=0, color_scheme_text="", **kwargs):
        config = load_config()
        api_key = config.get("api_key", "")
        url = config.get("api_url", "https://vip.dmxapi.com/v1/chat/completions")
        image_model = config.get("image_model", "gemini-2.5-flash-preview")
        
        print(f"[Renderer] ========== 输入参数 DEBUG ==========")
        print(f"[Renderer] visual_schema 长度: {len(visual_schema)} chars")
        print(f"[Renderer] visual_schema 前200字符:\n{visual_schema[:200]}")
        print(f"[Renderer] color_scheme_text: '{color_scheme_text[:100] if color_scheme_text else '(空)'}'")
        print(f"[Renderer] =====================================")
        
        if not api_key:
            return (create_error_image(), "错误: 未配置 API Key")
        
        schema_content = visual_schema
        if "---BEGIN PROMPT---" in visual_schema and "---END PROMPT---" in visual_schema:
            start = visual_schema.find("---BEGIN PROMPT---") + len("---BEGIN PROMPT---")
            end = visual_schema.find("---END PROMPT---")
            schema_content = visual_schema[start:end].strip()
            print(f"[Renderer] 检测到 BEGIN/END PROMPT 标记，提取内容长度: {len(schema_content)}")
        else:
            print(f"[Renderer] 未检测到 BEGIN/END PROMPT 标记，使用原始 visual_schema")
        
        max_schema_len = 4000
        if len(schema_content) > max_schema_len:
            print(f"[Renderer] Schema 过长 ({len(schema_content)} chars)，截断至 {max_schema_len}")
            schema_content = schema_content[:max_schema_len] + "\n...(truncated)"
        
        print(f"[Renderer] schema_content 前300字符:\n{schema_content[:300]}")
        
        render_prompt = f"""Generate a professional academic diagram based on this description:

{schema_content}

Style requirements:
- Flat vector graphics, clean lines
- Professional academic paper style
- Clean white background
- No 3D effects or shadows"""
        
        # 如果有预设颜色方案，强制指定
        if color_scheme_text:
            render_prompt += f"\n\n**CRITICAL COLOR INSTRUCTIONS (MUST FOLLOW):**\n{color_scheme_text}"
            print(f"[Renderer] 添加了预设颜色方案到 prompt")
        
        content = []
        
        max_ref_images = 2
        ref_count = 0
        for i in range(1, 9):
            if ref_count >= max_ref_images:
                break
            ref_img = kwargs.get(f"ref_image_{i}")
            if ref_img is not None:
                img_base64 = image_to_base64(ref_img, format="JPEG", max_size=800, quality=85)
                if img_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
                    ref_count += 1
        
        if ref_count > 0:
            render_prompt += f"\n- Match the visual style of the {ref_count} reference image(s)"
        
        content.append({
            "type": "text",
            "text": render_prompt
        })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        payload = {
            "model": image_model,
            "messages": [{"role": "user", "content": content}],
        }
        
        info_text = "Renderer 执行中..."
        
        print(f"[Renderer] ========== 请求详情 ==========")
        print(f"[Renderer] URL: {url}")
        print(f"[Renderer] Model: {image_model}")
        print(f"[Renderer] API Key: {api_key[:10]}...{api_key[-4:]}")
        print(f"[Renderer] Prompt 长度: {len(render_prompt)} chars")
        print(f"[Renderer] 参考图数量: {ref_count}")
        print(f"[Renderer] ========== 完整 render_prompt ==========")
        print(render_prompt)
        print(f"[Renderer] =========================================")
        
        try:
            print("[Renderer] 正在发送请求...")
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            print(f"[Renderer] 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                resp_content = result["choices"][0]["message"]["content"]
                
                image_bytes = parse_image_from_response(resp_content)
                if image_bytes:
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    np_image = np.array(pil_image).astype(np.float32) / 255.0
                    info_text = f"渲染成功: {pil_image.size}"
                    print(f"[Renderer] {info_text}")
                    return (torch.from_numpy(np_image).unsqueeze(0), info_text)
                else:
                    info_text = f"返回文本: {resp_content[:200]}"
            else:
                error_detail = response.text[:500] if response.text else "无详细信息"
                info_text = f"API 错误: {response.status_code} - {error_detail}"
                print(f"[Renderer] {info_text}")
                
        except Exception as e:
            info_text = f"错误: {str(e)}"
        
        print(f"[Renderer] {info_text}")
        return (create_error_image(), info_text)
