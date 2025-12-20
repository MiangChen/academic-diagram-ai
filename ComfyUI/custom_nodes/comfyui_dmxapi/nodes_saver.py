"""
BatchImageSaver - 批量保存图像
"""

import os
import numpy as np
import torch
from PIL import Image


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
        os.makedirs(output_folder, exist_ok=True)
        
        saved_paths = []
        
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            batch_size = images.shape[0]
            print(f"[BatchImageSaver] 保存 {batch_size} 张图像到 {output_folder}, auto_crop={auto_crop}")
            
            for i in range(batch_size):
                img_tensor = images[i]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                if len(img_np.shape) == 2:
                    pil_img = Image.fromarray(img_np, mode='L')
                elif img_np.shape[2] == 3:
                    pil_img = Image.fromarray(img_np, mode='RGB')
                elif img_np.shape[2] == 4:
                    pil_img = Image.fromarray(img_np, mode='RGBA')
                    
                    if auto_crop:
                        alpha = img_np[:, :, 3]
                        rows = np.any(alpha > 0, axis=1)
                        cols = np.any(alpha > 0, axis=0)
                        
                        if np.any(rows) and np.any(cols):
                            y_indices = np.where(rows)[0]
                            x_indices = np.where(cols)[0]
                            y_min, y_max = y_indices[0], y_indices[-1]
                            x_min, x_max = x_indices[0], x_indices[-1]
                            
                            pil_img = pil_img.crop((x_min, y_min, x_max + 1, y_max + 1))
                else:
                    print(f"[BatchImageSaver] 跳过未知格式: shape={img_np.shape}")
                    continue
                
                filename = f"{filename_prefix}_{i:04d}.png"
                filepath = os.path.join(output_folder, filename)
                
                pil_img.save(filepath, 'PNG')
                saved_paths.append(filepath)
                print(f"[BatchImageSaver] 已保存: {filepath} ({pil_img.size[0]}x{pil_img.size[1]})")
        
        result = f"已保存 {len(saved_paths)} 张图像:\n" + "\n".join(saved_paths)
        return (result,)
