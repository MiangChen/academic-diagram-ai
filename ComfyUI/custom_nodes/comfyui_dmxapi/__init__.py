"""
ComfyUI 自定义节点 - 学术插图工作流
"""

from .nodes_architect import AcademicArchitect
from .nodes_renderer import AcademicRenderer
from .nodes_editor import AcademicEditor
from .nodes_segmenter import ColorRegionSegmenter, TransparentSplitter
from .nodes_saver import BatchImageSaver

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AcademicArchitect": AcademicArchitect,
    "AcademicRenderer": AcademicRenderer,
    "AcademicEditor": AcademicEditor,
    "ColorRegionSegmenter": ColorRegionSegmenter,
    "TransparentSplitter": TransparentSplitter,
    "BatchImageSaver": BatchImageSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AcademicArchitect": "学术插图 - 逻辑构建 (Architect)",
    "AcademicRenderer": "学术插图 - 视觉渲染 (Renderer)",
    "AcademicEditor": "学术插图 - 交互编辑 (Editor)",
    "ColorRegionSegmenter": "学术插图 - 颜色区域分割 (ColorSegmenter)",
    "TransparentSplitter": "学术插图 - 透明分割 (TransparentSplitter)",
    "BatchImageSaver": "学术插图 - 批量保存 (BatchSaver)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
