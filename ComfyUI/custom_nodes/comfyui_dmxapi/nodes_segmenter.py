"""
图像分割节点 - ColorRegionSegmenter, TransparentSplitter
"""

import numpy as np
import torch


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
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 洪水填充函数
        def flood_fill_by_color(start_y, start_x, visited):
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
                        color_diff = np.max(np.abs(neighbor_color - current_color))
                        
                        if color_diff <= color_tolerance:
                            visited[ny, nx] = True
                            region_mask[ny, nx] = True
                            queue.append((ny, nx))
                            area += 1
            
            return region_mask, area
        
        visited = np.zeros((h, w), dtype=bool)
        candidate_regions = []
        
        print(f"[ColorSegmenter] 开始扫描所有像素...")
        
        for y in range(h):
            for x in range(w):
                if visited[y, x]:
                    continue
                
                region_mask, area = flood_fill_by_color(y, x, visited)
                
                if region_mask is not None and min_area <= area <= max_area:
                    candidate_regions.append((region_mask, area))
                    print(f"[ColorSegmenter] 找到候选区域: {area} 像素 ({area/total_pixels*100:.2f}%)")
        
        print(f"[ColorSegmenter] 候选区域数: {len(candidate_regions)}")
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        valid_regions = 0
        
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
        
        mask_tensor = torch.from_numpy(final_mask.astype(np.float32) / 255.0).unsqueeze(0)
        segmented_tensor = torch.from_numpy(segmented_vis.astype(np.float32) / 255.0).unsqueeze(0)
        
        # 创建 masked_image：mask 区域变透明
        masked_vis = np.zeros((h, w, 4), dtype=np.uint8)
        masked_vis[:, :, :3] = img_np
        masked_vis[:, :, 3] = 255
        mask_bool = final_mask > 0
        masked_vis[mask_bool, 3] = 0
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
        
        if channels == 4:
            alpha = img_np[:, :, 3]
        else:
            if channels == 3:
                white_mask = np.all(img_np > 250, axis=2)
                alpha = np.where(white_mask, 0, 255).astype(np.uint8)
            else:
                alpha = np.where(img_np > 250, 0, 255).astype(np.uint8)
        
        binary = (alpha >= alpha_threshold).astype(np.int32)
        
        if use_8_connectivity:
            structure = np.ones((3, 3), dtype=np.int32)
        else:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
        
        labeled, num_features = ndimage.label(binary, structure=structure)
        print(f"[TransparentSplitter] 找到 {num_features} 个连通区域")
        
        valid_regions = []
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            area = np.sum(region_mask)
            
            if area >= min_area:
                rows = np.any(region_mask, axis=1)
                cols = np.any(region_mask, axis=0)
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                
                if len(y_indices) > 0 and len(x_indices) > 0:
                    y_min, y_max = y_indices[0], y_indices[-1]
                    x_min, x_max = x_indices[0], x_indices[-1]
                    
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
            info = "未找到有效区域"
            img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
            return (img_tensor, 0, info)
        
        cropped_images = []
        
        if output_mode == "original":
            for region in valid_regions:
                output = np.zeros((h, w, 4), dtype=np.uint8)
                output[:, :, :3] = img_np[:, :, :3] if channels >= 3 else np.stack([img_np]*3, axis=2)
                output[:, :, 3] = np.where(region['mask'], alpha, 0)
                cropped_images.append(output)
        
        elif output_mode == "crop":
            for region in valid_regions:
                y_min, y_max, x_min, x_max = region['bbox']
                crop_h = y_max - y_min + 1
                crop_w = x_max - x_min + 1
                
                output = np.zeros((crop_h, crop_w, 4), dtype=np.uint8)
                
                if channels >= 3:
                    output[:, :, :3] = img_np[y_min:y_max+1, x_min:x_max+1, :3]
                else:
                    gray = img_np[y_min:y_max+1, x_min:x_max+1]
                    output[:, :, :3] = np.stack([gray]*3, axis=2)
                
                region_crop = region['mask'][y_min:y_max+1, x_min:x_max+1]
                alpha_crop = alpha[y_min:y_max+1, x_min:x_max+1]
                output[:, :, 3] = np.where(region_crop, alpha_crop, 0)
                
                cropped_images.append(output)
        
        elif output_mode == "uniform":
            max_h = max(r['bbox'][1] - r['bbox'][0] + 1 for r in valid_regions)
            max_w = max(r['bbox'][3] - r['bbox'][2] + 1 for r in valid_regions)
            
            for region in valid_regions:
                y_min, y_max, x_min, x_max = region['bbox']
                crop_h = y_max - y_min + 1
                crop_w = x_max - x_min + 1
                
                output = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                
                offset_y = (max_h - crop_h) // 2
                offset_x = (max_w - crop_w) // 2
                
                if channels >= 3:
                    output[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w, :3] = img_np[y_min:y_max+1, x_min:x_max+1, :3]
                else:
                    gray = img_np[y_min:y_max+1, x_min:x_max+1]
                    output[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w, :3] = np.stack([gray]*3, axis=2)
                
                region_crop = region['mask'][y_min:y_max+1, x_min:x_max+1]
                alpha_crop = alpha[y_min:y_max+1, x_min:x_max+1]
                output[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w, 3] = np.where(region_crop, alpha_crop, 0)
                
                cropped_images.append(output)
        
        if output_mode == "crop" and len(set((img.shape[0], img.shape[1]) for img in cropped_images)) > 1:
            max_h = max(img.shape[0] for img in cropped_images)
            max_w = max(img.shape[1] for img in cropped_images)
            
            padded_images = []
            for img in cropped_images:
                padded = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                padded[:img.shape[0], :img.shape[1]] = img
                padded_images.append(padded)
            cropped_images = padded_images
        
        tensors = [torch.from_numpy(img.astype(np.float32) / 255.0) for img in cropped_images]
        batch_tensor = torch.stack(tensors, dim=0)
        
        info = f"分割出 {len(valid_regions)} 个元件 (模式: {output_mode})"
        
        return (batch_tensor, len(valid_regions), info)
