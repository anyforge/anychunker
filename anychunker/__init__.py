import importlib
from typing import TYPE_CHECKING

# modules class mapping
_LAZY_IMPORT_MAPPING = {
    # base
    "BaseTextChunker": ".base",
    "Language": ".base",
    "AnySeparators": ".base",
    
    # schemas
    "AnyDataModel": ".schemas",
    "Chunker": ".schemas",
    "DocumentMetadata": ".schemas",
    "ChunkBatcher": ".schemas",
    
    # text
    "AnyTextChunker": ".text",
    
    # markdown
    "AnyMarkdownChunker": ".markdown",
    
    # code
    "AnyCodeChunker": ".code",
    
    # sentence
    "AnySentenceChunker": ".sentence",
    
    # semantics
    "AnySemanticsChunker": ".semantics",
    
    # blocks
    "AnyMarkdownBlockChunker": ".blocks",
 
    # version
    "__version__": ".__version__"
}


__all__ = list(_LAZY_IMPORT_MAPPING.keys())


# module __getattr__
def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAPPING:
        module_path = _LAZY_IMPORT_MAPPING[name]
        # 动态导入对应的子模块
        module = importlib.import_module(module_path, package=__name__)
        # 从子模块中提取对应的类/对象
        attr = getattr(module, name)
        # 将其缓存到当前模块的全局变量中，避免下次重复 import
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 4. [IDE 提示支持] 仅供静态类型检查器（如 mypy, PyCharm）使用
if TYPE_CHECKING:
    from .base import BaseTextChunker, Language, AnySeparators
    from .schemas import (
        AnyDataModel,
        Chunker,
        DocumentMetadata,
        ChunkBatcher,
    )
    from .text import AnyTextChunker
    from .markdown import AnyMarkdownChunker
    from .code import AnyCodeChunker
    from .sentence import AnySentenceChunker
    from .semantics import AnySemanticsChunker
    from .blocks import AnyMarkdownBlockChunker
    
    