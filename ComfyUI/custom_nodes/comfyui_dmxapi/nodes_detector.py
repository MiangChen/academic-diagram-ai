"""
图标检测和物体定位节点
"""

import requests
import torch

from .config import load_config
from .utils import image_to_base64


class AcademicIconDetector:
    """
    图标检测器 - 分析图像中的所有图标/元素，输出列表供 SAM2/GroundingDino 使用
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "custom_categories": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("icon_list", "detailed_info")
    FUNCTION = "detect"
    CATEGORY = "DMXAPI/Academic"

    def detect(self, image, custom_categories=""):
        config = load_config()
        api_key = config.get("api_key", "")
        url = config.get("api_url", "https://vip.dmxapi.com/v1/chat/completions")
        text_model = config.get("text_model", "gemini-2.5-flash-preview")
        
        if not api_key:
            return ("error", "错误: 未配置 API Key")
        
        img_base64 = image_to_base64(image)
        
        detect_prompt = """Analyze this academic diagram and list ALL visual icons/objects that could be extracted as individual assets.

For each icon, provide a simple English label that could be used for object detection.

Output format - just a comma-separated list of object names, like:
drone, tank, robot dog, humanoid robot, building, gear icon, gamepad, helicopter

Rules:
- Only list actual visual objects/icons, not text labels or arrows
- Use simple, common English words
- Each item should be a distinct visual element that could be cut out
- Do not include: boxes, frames, backgrounds, lines, arrows, text"""

        if custom_categories:
            detect_prompt += f"\n\nFocus especially on these categories: {custom_categories}"
        
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            {"type": "text", "text": detect_prompt}
        ]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        payload = {
            "model": text_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
        }
        
        try:
            print(f"[IconDetector] 使用模型: {text_model}")
            print("[IconDetector] 正在分析图像中的图标...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                icon_list = result["choices"][0]["message"]["content"].strip()
                icon_list = icon_list.replace("\n", ", ").replace("  ", " ")
                print(f"[IconDetector] 检测到图标: {icon_list}")
                return (icon_list, f"检测成功，找到图标: {icon_list}")
            else:
                error = f"API 错误: {response.status_code}"
                print(f"[IconDetector] {error}")
                return ("", error)
                
        except Exception as e:
            error = f"错误: {str(e)}"
            print(f"[IconDetector] {error}")
            return ("", error)


class AcademicObjectLocator:
    """
    物体定位器 - 用 LLM 分析图像，返回指定物体的坐标，可直接连接 SAM2
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "object_prompt": ("STRING", {
                    "multiline": True,
                    "default": "drone, tank, robot dog, humanoid robot"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("coordinates", "info")
    FUNCTION = "locate"
    CATEGORY = "DMXAPI/Academic"

    def locate(self, image, object_prompt):
        import json
        
        config = load_config()
        api_key = config.get("api_key", "")
        url = config.get("api_url", "https://vip.dmxapi.com/v1/chat/completions")
        text_model = config.get("text_model", "gemini-2.5-flash-preview")
        
        if not api_key:
            return ("[]", "错误: 未配置 API Key")
        
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                h, w = image.shape[1], image.shape[2]
            else:
                h, w = image.shape[0], image.shape[1]
        else:
            h, w = image.shape[0], image.shape[1]
        
        img_base64 = image_to_base64(image)
        
        locate_prompt = f"""Analyze this image and find the CENTER POINT coordinates of each object I specify.

Image size: {w} x {h} pixels

Objects to find: {object_prompt}

For EACH object found, provide its center point as x,y coordinates.

IMPORTANT: Output ONLY a valid JSON array in this exact format, nothing else:
[{{"x": 123, "y": 456}}, {{"x": 789, "y": 101}}]

Rules:
- x is horizontal position (0 = left edge, {w} = right edge)
- y is vertical position (0 = top edge, {h} = bottom edge)
- Find ALL instances of the specified objects
- If an object appears multiple times, include each instance
- Output ONLY the JSON array, no explanation"""

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
            {"type": "text", "text": locate_prompt}
        ]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        payload = {
            "model": text_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
        }
        
        try:
            print(f"[ObjectLocator] 使用模型: {text_model}")
            print(f"[ObjectLocator] 图像尺寸: {w}x{h}")
            print(f"[ObjectLocator] 查找物体: {object_prompt}")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                raw_output = result["choices"][0]["message"]["content"].strip()
                print(f"[ObjectLocator] 原始输出: {raw_output}")
                
                try:
                    if "```" in raw_output:
                        import re
                        json_match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
                        if json_match:
                            raw_output = json_match.group()
                    
                    coords = json.loads(raw_output)
                    
                    if not coords or len(coords) == 0:
                        coords = [{"x": w // 2, "y": h // 2}]
                        info = "未找到指定物体，返回图像中心坐标"
                        print(f"[ObjectLocator] {info}")
                        return (json.dumps(coords), info)
                    
                    coords_str = json.dumps(coords)
                    info = f"找到 {len(coords)} 个物体坐标"
                    print(f"[ObjectLocator] {info}: {coords_str}")
                    return (coords_str, info)
                except json.JSONDecodeError as e:
                    print(f"[ObjectLocator] JSON 解析失败: {e}")
                    default_coords = [{"x": w // 2, "y": h // 2}]
                    return (json.dumps(default_coords), f"JSON 解析失败，返回默认坐标: {raw_output[:100]}")
            else:
                error = f"API 错误: {response.status_code}"
                print(f"[ObjectLocator] {error}")
                default_coords = [{"x": w // 2, "y": h // 2}]
                return (json.dumps(default_coords), error)
                
        except Exception as e:
            error = f"错误: {str(e)}"
            print(f"[ObjectLocator] {error}")
            return ('[{"x": 100, "y": 100}]', error)
