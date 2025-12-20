"""
学术插图工作流节点
"""

import requests
import numpy as np
import torch
from PIL import Image
import io
import base64

from .config import load_config
from .constants import LAYOUT_TYPES
from .prompts import ARCHITECT_PROMPT, RENDERER_PROMPT
from .utils import image_to_base64, parse_image_from_response, bytes_to_tensor, create_error_image


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
                "color_palette": ("STRING", {
                    "default": "Azure Blue (#E1F5FE), Slate Grey (#607D8B), Coral Orange (#FF7043), Mint Green (#A5D6A7)"
                }),
            },
            "optional": image_inputs
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "render"
    CATEGORY = "DMXAPI/Academic"

    def render(self, visual_schema, color_palette="", **kwargs):
        config = load_config()
        api_key = config.get("api_key", "")
        url = config.get("api_url", "https://vip.dmxapi.com/v1/chat/completions")
        image_model = config.get("image_model", "gemini-2.5-flash-preview")
        
        # ========== DEBUG LOG ==========
        print(f"[Renderer] ========== 输入参数 DEBUG ==========")
        print(f"[Renderer] visual_schema 长度: {len(visual_schema)} chars")
        print(f"[Renderer] visual_schema 前200字符:\n{visual_schema[:200]}")
        print(f"[Renderer] color_palette: '{color_palette}'")
        print(f"[Renderer] =====================================")
        # ================================
        
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
        
        if color_palette:
            render_prompt += f"\n- Colors: {color_palette}"
            print(f"[Renderer] 添加了 color_palette 到 prompt")
        
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
                # 清理输出，确保是简单的逗号分隔列表
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
        
        # 获取图像尺寸
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
                
                # 尝试提取 JSON
                try:
                    # 清理可能的 markdown 代码块
                    if "```" in raw_output:
                        import re
                        json_match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
                        if json_match:
                            raw_output = json_match.group()
                    
                    coords = json.loads(raw_output)
                    
                    # 确保坐标不为空
                    if not coords or len(coords) == 0:
                        # 返回图像中心作为默认坐标
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
                    # 返回图像中心作为默认坐标
                    default_coords = [{"x": w // 2, "y": h // 2}]
                    return (json.dumps(default_coords), f"JSON 解析失败，返回默认坐标: {raw_output[:100]}")
            else:
                error = f"API 错误: {response.status_code}"
                print(f"[ObjectLocator] {error}")
                # 返回图像中心作为默认坐标
                default_coords = [{"x": w // 2, "y": h // 2}]
                return (json.dumps(default_coords), error)
                
        except Exception as e:
            error = f"错误: {str(e)}"
            print(f"[ObjectLocator] {error}")
            # 返回默认坐标
            return ('[{"x": 100, "y": 100}]', error)


class ColorRegionSegmenter:
    """
    颜色区域分割器 - 基于洪水填充的颜色连通区域分割
    通过颜色相似度扩散，自动识别边界
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_area_percent": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 50.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "max_area_percent": ("FLOAT", {
                    "default": 15.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "color_tolerance": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
                "use_8_connectivity": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "segmented_image", "masked_image", "info")
    FUNCTION = "segment"
    CATEGORY = "DMXAPI/Academic"

    def segment(self, image, min_area_percent, max_area_percent, 
                color_tolerance, use_8_connectivity):
        import cv2
        from collections import deque
        
        # 转换图像格式
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                img_np = image[0].cpu().numpy()
            else:
                img_np = image.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = (image * 255).astype(np.uint8)
        
        h, w = img_np.shape[:2]
        total_pixels = h * w
        min_area = int(total_pixels * min_area_percent / 100)
        max_area = int(total_pixels * max_area_percent / 100)
        
        connectivity_str = "8-连通" if use_8_connectivity else "4-连通"
        print(f"[ColorSegmenter] 图像尺寸: {w}x{h}, 总像素: {total_pixels}")
        print(f"[ColorSegmenter] 面积范围: {min_area} - {max_area} 像素")
        print(f"[ColorSegmenter] 颜色容差: {color_tolerance}, 连通方式: {connectivity_str}")
        
        # 确保是3通道图像
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        # 定义邻居方向
        if use_8_connectivity:
            # 8-连通：上下左右 + 对角线
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            # 4-连通：只有上下左右
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 洪水填充函数 - 基于颜色相似度扩散
        def flood_fill_by_color(start_y, start_x, visited):
            """从起始点开始，基于颜色相似度进行洪水填充"""
            if visited[start_y, start_x]:
                return None, 0
            
            region_mask = np.zeros((h, w), dtype=bool)
            queue = deque([(start_y, start_x)])
            region_mask[start_y, start_x] = True
            visited[start_y, start_x] = True
            area = 1
            
            while queue:
                cy, cx = queue.popleft()
                current_color = img_np[cy, cx].astype(np.int32)
                
                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx
                    
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        neighbor_color = img_np[ny, nx].astype(np.int32)
                        
                        # 检查颜色差异 - 与当前像素比较
                        color_diff = np.max(np.abs(neighbor_color - current_color))
                        
                        if color_diff <= color_tolerance:
                            # 颜色相近，继续扩散
                            visited[ny, nx] = True
                            region_mask[ny, nx] = True
                            queue.append((ny, nx))
                            area += 1
            
            return region_mask, area
        
        # 全局访问标记
        visited = np.zeros((h, w), dtype=bool)
        
        # 收集所有候选区域
        candidate_regions = []
        
        print(f"[ColorSegmenter] 开始扫描所有像素...")
        
        # 遍历所有像素作为潜在种子点
        for y in range(h):
            for x in range(w):
                if visited[y, x]:
                    continue
                
                region_mask, area = flood_fill_by_color(y, x, visited)
                
                if region_mask is not None and min_area <= area <= max_area:
                    candidate_regions.append((region_mask, area))
                    print(f"[ColorSegmenter] 找到候选区域: {area} 像素 ({area/total_pixels*100:.2f}%)")
        
        print(f"[ColorSegmenter] 候选区域数: {len(candidate_regions)}")
        
        # 创建最终 mask
        final_mask = np.zeros((h, w), dtype=np.uint8)
        valid_regions = 0
        
        # 为可视化创建彩色分割图
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
            [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128],
            [255, 128, 0], [128, 255, 0], [0, 128, 255], [255, 0, 128]
        ]
        segmented_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for region_mask, area in candidate_regions:
            final_mask[region_mask] = 255
            color = colors[valid_regions % len(colors)]
            segmented_vis[region_mask] = color
            valid_regions += 1
            print(f"[ColorSegmenter] 区域 {valid_regions}: {area} 像素 ({area/total_pixels*100:.2f}%) ✓")
        
        print(f"[ColorSegmenter] 有效区域数: {valid_regions}")
        
        # 转换为 ComfyUI 格式
        mask_tensor = torch.from_numpy(final_mask.astype(np.float32) / 255.0).unsqueeze(0)
        segmented_tensor = torch.from_numpy(segmented_vis.astype(np.float32) / 255.0).unsqueeze(0)
        
        # 创建 masked_image：mask 区域变透明，输出 RGBA
        # 创建 4 通道图像 (RGBA)
        masked_vis = np.zeros((h, w, 4), dtype=np.uint8)
        masked_vis[:, :, :3] = img_np  # RGB 通道复制原图
        masked_vis[:, :, 3] = 255  # Alpha 通道默认不透明
        mask_bool = final_mask > 0
        masked_vis[mask_bool, 3] = 0  # mask 区域设为透明
        masked_tensor = torch.from_numpy(masked_vis.astype(np.float32) / 255.0).unsqueeze(0)
        
        info = f"找到 {valid_regions} 个有效区域 (容差: {color_tolerance}, {connectivity_str})"
        
        return (mask_tensor, segmented_tensor, masked_tensor, info)



class TransparentSplitter:
    """
    透明区域分割器 - 基于透明度将 PNG 图像分割成多个独立元件
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_threshold": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 255,
                    "step": 1,
                    "display": "slider"
                }),
                "min_area": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 10000,
                    "step": 10,
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "use_8_connectivity": ("BOOLEAN", {"default": False}),
                "output_mode": (["crop", "original", "uniform"], {"default": "crop"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "count", "info")
    FUNCTION = "split"
    CATEGORY = "DMXAPI/Academic"

    def split(self, image, alpha_threshold, min_area, padding, use_8_connectivity, output_mode):
        from scipy import ndimage
        
        # 转换图像格式
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                img_np = image[0].cpu().numpy()
            else:
                img_np = image.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = (image * 255).astype(np.uint8)
        
        h, w = img_np.shape[:2]
        channels = img_np.shape[2] if len(img_np.shape) == 3 else 1
        
        print(f"[TransparentSplitter] 图像尺寸: {w}x{h}, 通道数: {channels}")
        
        # 获取 alpha 通道
        if channels == 4:
            alpha = img_np[:, :, 3]
        else:
            # 没有 alpha 通道，用白色作为透明
            if channels == 3:
                # 检测白色像素作为透明
                white_mask = np.all(img_np > 250, axis=2)
                alpha = np.where(white_mask, 0, 255).astype(np.uint8)
            else:
                alpha = np.where(img_np > 250, 0, 255).astype(np.uint8)
        
        # 二值化：非透明像素为 1
        binary = (alpha >= alpha_threshold).astype(np.int32)
        
        # 连通区域标记
        if use_8_connectivity:
            structure = np.ones((3, 3), dtype=np.int32)  # 8-连通
        else:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)  # 4-连通
        
        labeled, num_features = ndimage.label(binary, structure=structure)
        print(f"[TransparentSplitter] 找到 {num_features} 个连通区域")
        
        # 收集有效区域
        valid_regions = []
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            area = np.sum(region_mask)
            
            if area >= min_area:
                # 找到边界框
                rows = np.any(region_mask, axis=1)
                cols = np.any(region_mask, axis=0)
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                
                if len(y_indices) > 0 and len(x_indices) > 0:
                    y_min, y_max = y_indices[0], y_indices[-1]
                    x_min, x_max = x_indices[0], x_indices[-1]
                    
                    # 添加 padding
                    y_min = max(0, y_min - padding)
                    y_max = min(h - 1, y_max + padding)
                    x_min = max(0, x_min - padding)
                    x_max = min(w - 1, x_max + padding)
                    
                    valid_regions.append({
                        'mask': region_mask,
                        'bbox': (y_min, y_max, x_min, x_max),
                        'area': area
                    })
                    print(f"[TransparentSplitter] 区域 {len(valid_regions)}: {area} 像素, bbox=({x_min},{y_min})-({x_max},{y_max})")
        
        print(f"[TransparentSplitter] 有效区域数: {len(valid_regions)}")
        
        if len(valid_regions) == 0:
            # 没有找到有效区域，返回原图
            info = "未找到有效区域"
            img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
            return (img_tensor, 0, info)
        
        # 根据 output_mode 处理输出
        cropped_images = []
        
        if output_mode == "original":
            # 保持原尺寸，非当前区域变透明
            for region in valid_regions:
                output = np.zeros((h, w, 4), dtype=np.uint8)
                output[:, :, :3] = img_np[:, :, :3] if channels >= 3 else np.stack([img_np]*3, axis=2)
                output[:, :, 3] = np.where(region['mask'], alpha, 0)
                cropped_images.append(output)
        
        elif output_mode == "crop":
            # 裁剪到内容区域
            for region in valid_regions:
                y_min, y_max, x_min, x_max = region['bbox']
                crop_h = y_max - y_min + 1
                crop_w = x_max - x_min + 1
                
                output = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
                
                # 复制 RGB
                if channels >= 3:
                    output[:, :, :3] = img_np[y_min:y_max+1, x_min:x_max+1, :3]
                else:
                    gray = img_np[y_min:y_max+1, x_min:x_max+1]
                    output[:, :, :3] = np.stack([gray]*3, axis=2)
                
                # 设置 alpha：只有当前区域的像素不透明
                region_crop = region['mask'][y_min:y_max+1, x_min:x_max+1]
                alpha_crop = alpha[y_min:y_max+1, x_min:x_max+1]
                output[:, :, 3] = np.where(region_crop, alpha_crop, 0)
                
                cropped_images.append(output)
        
        elif output_mode == "uniform":
            # 统一到最大尺寸
            max_h = max(r['bbox'][1] - r['bbox'][0] + 1 for r in valid_regions)
            max_w = max(r['bbox'][3] - r['bbox'][2] + 1 for r in valid_regions)
            
            for region in valid_regions:
                y_min, y_max, x_min, x_max = region['bbox']
                crop_h = y_max - y_min + 1
                crop_w = x_max - x_min + 1
                
                # 创建统一尺寸的输出，内容居中
                output = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                
                # 计算居中偏移
                offset_y = (max_h - crop_h) // 2
                offset_x = (max_w - crop_w) // 2
                
                # 复制 RGB
                if channels >= 3:
                    output[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w, :3] = img_np[y_min:y_max+1, x_min:x_max+1, :3]
                else:
                    gray = img_np[y_min:y_max+1, x_min:x_max+1]
                    output[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w, :3] = np.stack([gray]*3, axis=2)
                
                # 设置 alpha
                region_crop = region['mask'][y_min:y_max+1, x_min:x_max+1]
                alpha_crop = alpha[y_min:y_max+1, x_min:x_max+1]
                output[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w, 3] = np.where(region_crop, alpha_crop, 0)
                
                cropped_images.append(output)
        
        # 转换为 tensor batch
        # 对于 crop 模式，不同尺寸的图像需要特殊处理
        if output_mode == "crop" and len(set((img.shape[0], img.shape[1]) for img in cropped_images)) > 1:
            # 不同尺寸，pad 到最大尺寸
            max_h = max(img.shape[0] for img in cropped_images)
            max_w = max(img.shape[1] for img in cropped_images)
            
            padded_images = []
            for img in cropped_images:
                padded = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                padded[:img.shape[0], :img.shape[1]] = img
                padded_images.append(padded)
            cropped_images = padded_images
        
        # 转换为 tensor
        tensors = [torch.from_numpy(img.astype(np.float32) / 255.0) for img in cropped_images]
        batch_tensor = torch.stack(tensors, dim=0)
        
        info = f"分割出 {len(valid_regions)} 个元件 (模式: {output_mode})"
        
        return (batch_tensor, len(valid_regions), info)



class BatchImageSaver:
    """
    批量图像保存器 - 保存 batch 图像为 PNG（支持透明通道）
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "split"}),
                "output_folder": ("STRING", {"default": "output/splits"}),
                "auto_crop": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_paths",)
    FUNCTION = "save"
    CATEGORY = "DMXAPI/Academic"
    OUTPUT_NODE = True

    def save(self, images, filename_prefix, output_folder, auto_crop):
        import os
        from PIL import Image
        
        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)
        
        saved_paths = []
        
        # 处理 batch
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            batch_size = images.shape[0]
            print(f"[BatchImageSaver] 保存 {batch_size} 张图像到 {output_folder}, auto_crop={auto_crop}")
            
            for i in range(batch_size):
                img_tensor = images[i]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                # 判断通道数
                if len(img_np.shape) == 2:
                    # 灰度图
                    pil_img = Image.fromarray(img_np, mode='L')
                elif img_np.shape[2] == 3:
                    # RGB
                    pil_img = Image.fromarray(img_np, mode='RGB')
                elif img_np.shape[2] == 4:
                    # RGBA
                    pil_img = Image.fromarray(img_np, mode='RGBA')
                    
                    # 自动裁剪透明边界
                    if auto_crop:
                        alpha = img_np[:, :, 3]
                        rows = np.any(alpha > 0, axis=1)
                        cols = np.any(alpha > 0, axis=0)
                        
                        if np.any(rows) and np.any(cols):
                            y_indices = np.where(rows)[0]
                            x_indices = np.where(cols)[0]
                            y_min, y_max = y_indices[0], y_indices[-1]
                            x_min, x_max = x_indices[0], x_indices[-1]
                            
                            # 裁剪
                            pil_img = pil_img.crop((x_min, y_min, x_max + 1, y_max + 1))
                else:
                    print(f"[BatchImageSaver] 跳过未知格式: shape={img_np.shape}")
                    continue
                
                # 生成文件名
                filename = f"{filename_prefix}_{i:04d}.png"
                filepath = os.path.join(output_folder, filename)
                
                # 保存
                pil_img.save(filepath, 'PNG')
                saved_paths.append(filepath)
                print(f"[BatchImageSaver] 已保存: {filepath} ({pil_img.size[0]}x{pil_img.size[1]})")
        
        result = f"已保存 {len(saved_paths)} 张图像:\n" + "\n".join(saved_paths)
        return (result,)
