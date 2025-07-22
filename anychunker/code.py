import re
from typing import Optional,Callable,List, Union, Literal
from .base import Language
from .text import AnyTextChunker


class AnyCodeChunker(AnyTextChunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @classmethod
    def from_html(cls, **kwargs):
        return super().from_language(Language.HTML, **kwargs)
    
    @classmethod
    def from_markdown(cls, **kwargs):
        return super().from_language(Language.MARKDOWN, **kwargs)
    
    @classmethod
    def from_c(cls, **kwargs):
        return super().from_language(Language.C, **kwargs)
    
    @classmethod
    def from_python(cls, **kwargs):
        return super().from_language(Language.PYTHON, **kwargs)
        
    
    
    