import abc
import json
import datetime
from enum import Enum
from pydantic import BaseModel,Field
from typing import Optional, Iterator, Generator, Callable, Literal, Any , Any, Dict, List, Set


class AnyDataModel(BaseModel):
    class Config:
        extra = 'allow'  # 允许额外的字段
        use_enum_values = True # 序列化时使用枚举的实际值，而不是枚举对象本身。
        arbitrary_types_allowed = True #允许 Pydantic 处理任意类型
        arbitrary_types_allowed = True
        # 配置JSON编码器
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat(),
            datetime.time: lambda v: v.isoformat(),
        }
        
    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'

    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, key: str):
        """
        支持 doc['key'] 的取值方式
        
        Args:
            key: 字段名
            
        Returns:
            字段值
            
        Raises:
            KeyError: 当字段不存在时
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"字段 '{key}' 不存在")
    
    def __setitem__(self, key: str, value: Any):
        """
        支持 doc['key'] = value 的设值方式
        
        Args:
            key: 字段名
            value: 字段值
        """
        setattr(self, key, value)
    
    def __delitem__(self, key: str):
        """
        支持 del doc['key'] 的删除方式
        
        Args:
            key: 字段名
            
        Raises:
            KeyError: 当字段不存在时
        """
        if hasattr(self, key):
            delattr(self, key)
        else:
            raise KeyError(f"字段 '{key}' 不存在")
    
    def __contains__(self, key: str):
        """
        支持 'key' in doc 的检查方式
        
        Args:
            key: 字段名
            
        Returns:
            是否包含该字段
        """
        return hasattr(self, key)
    
    def __len__(self):
        """
        支持 len(doc) 获取字段数量
        
        Returns:
            字段数量
        """
        return len(self.model_fields_set) + len(self.model_extra or {})
    
    def __iter__(self):
        """
        支持 for key in doc 的迭代方式
        
        Returns:
            字段名迭代器
        """
        return iter(self.keys())

    def __eq__(self, other: Any):
        """
        实现相等性比较 (==)
        
        比较规则：
        1. 类型必须相同或兼容
        2. 所有字段值必须相等（忽略 created 时间）
        3. created 时间被自动忽略，因为它是自动生成的
        
        Args:
            other: 要比较的对象
            
        Returns:
            bool: 是否相等
        """
        if not isinstance(other, self.__class__):
            return False
        
        # 获取两个对象的所有字段数据
        self_data = self.model_dump()
        other_data = other.model_dump()
        
        # 移除 created 字段进行比较
        self_data.pop('created', None)
        other_data.pop('created', None)
        
        return self_data == other_data
    
    def __ne__(self, other: Any):
        """
        实现不等性比较 (!=)
        
        Args:
            other: 要比较的对象
            
        Returns:
            bool: 是否不相等
        """
        return not self.__eq__(other)
    
    def __hash__(self):
        """
        实现哈希方法，使对象可以用作字典键或集合元素
        
        注意：哈希计算时忽略 created 字段和其他不可变字段
        
        Returns:
            int: 哈希值
        """
        # 获取所有字段，但排除 created 字段
        data = self.model_dump()
        data.pop('created', None)  # 移除 created 字段
        
        # 只使用不可变字段计算哈希
        hashable_fields = []
        
        for field_name, field_value in data.items():
            if self._is_hashable(field_value):
                hashable_fields.append((field_name, field_value))
            else:
                # 对于不可哈希的字段，使用其字符串表示
                hashable_fields.append((field_name, str(field_value)))
        
        return hash(tuple(hashable_fields))
    
    def _is_hashable(self, obj: Any):
        """检查对象是否可哈希"""
        try:
            hash(obj)
            return True
        except TypeError:
            return False
    
    def equals(self, other, 
               ignore_fields: Set[str] = None,
               ignore_created: bool = True):
        """
        增强的相等性比较
        
        Args:
            other: 要比较的对象
            ignore_fields: 要忽略的字段集合
            ignore_created: 是否忽略创建时间
            deep: 是否深度比较
            
        Returns:
            bool: 是否相等
        """
        if not isinstance(other, self.__class__):
            return False
        
        # 准备忽略的字段集合
        ignore_set = set(ignore_fields or [])
        if ignore_created:
            ignore_set.add('created')
        
        # 获取数据
        self_data = self.model_dump()
        other_data = other.model_dump()
        
        # 移除忽略的字段
        for field in ignore_set:
            self_data.pop(field, None)
            other_data.pop(field, None)
        
        return self_data == other_data
    
    def keys(self):
        """
        获取所有字段名
        
        Returns:
            字段名的视图
        """
        data = self.model_dump()
        return data.keys()
    
    def values(self):
        """
        获取所有字段值
        
        Returns:
            字段值的视图
        """
        data = self.model_dump()
        return data.values()
    
    def items(self):
        """
        获取所有字段名值对
        
        Returns:
            字段名值对的视图
        """
        data = self.model_dump()
        return data.items()
    
    def get(self, key: str, default: Any = None):
        """
        安全获取字段值
        
        Args:
            key: 字段名
            default: 默认值
            
        Returns:
            字段值或默认值
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def pop(self, key: str, default: Any = None):
        """
        弹出字段值（删除并返回）
        
        Args:
            key: 字段名
            default: 默认值（当字段不存在时）
            
        Returns:
            字段值
            
        Raises:
            KeyError: 当字段不存在且未提供默认值时
        """
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is None and len([arg for arg in [default] if arg is not None]) == 0:
                # 没有提供默认值参数
                raise
            return default
    
    def update(self, *args, **kwargs):
        """
        更新字段值
        
        支持多种调用方式：
        - doc.update({"key": "value"})
        - doc.update(key="value")
        - doc.update({"key1": "value1"}, key2="value2")
        
        Args:
            *args: 字典或可迭代对象
            **kwargs: 关键字参数
        """
        # 处理位置参数
        for arg in args:
            if hasattr(arg, 'keys'):
                # 字典类型
                for key in arg:
                    self[key] = arg[key]
            else:
                # 可迭代对象，包含键值对
                for key, value in arg:
                    self[key] = value
        
        # 处理关键字参数
        for key, value in kwargs.items():
            self[key] = value
    
    def setdefault(self, key: str, default: Any = None):
        """
        设置默认值（如果字段不存在）
        
        Args:
            key: 字段名
            default: 默认值
            
        Returns:
            字段值
        """
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default
    
    def clear(self):
        """
        清空所有字段（保留模型定义的字段）
        """
        # 获取所有字段名
        all_keys = list(self.keys())
        
        # 删除所有额外字段
        for key in all_keys:
            if key not in self.keys():
                try:
                    del self[key]
                except (KeyError, AttributeError):
                    pass
        
        # 重置模型定义的字段为默认值
        for field_name, field_info in self.model_fields().items():
            if field_info.default is not None:
                self[field_name] = field_info.default
            elif field_info.default_factory is not None:
                self[field_name] = field_info.default_factory()
                
    # 增强的复制方法
    def copy(self, deep: bool = True, **updates):
        """
        增强的复制方法，支持同时更新字段
        
        Args:
            deep: 是否深复制
            **updates: 要更新的字段
            
        Returns:
            Document: 复制的新实例
        """
        return self.model_copy(deep=deep, update=updates)

    @classmethod
    def model_fields(cls):
        return getattr(cls, '__pydantic_fields__', {})

    def to_dict(self, mode: Literal['json', 'python'] | str = 'python', **kwargs):
        """转换为字典"""
        return self.model_dump(mode = mode, **kwargs)

    def to_json(self, **kwargs):
        """转换为JSON字符串"""
        return self.model_dump_json(**kwargs)

    def save_json(self, filename: str, encoding: str = 'utf-8', indent: int = 4, ensure_ascii: bool = False, **kwargs):
        """保存为 JSON 文件"""
        with open(filename, 'w', encoding=encoding) as f:
            json.dump(self.to_dict(mode = 'json'), f, indent = indent, ensure_ascii=ensure_ascii, **kwargs)
        return

    @classmethod
    def load_from_json(cls, filename: str, encoding: str = 'utf-8'):
        """从 JSON 文件加载"""
        with open(filename, 'r', encoding = encoding) as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]):
        """从字典加载"""
        return cls.model_validate(data)


class Chunker(AnyDataModel):
    metadata: Dict[str, Any] = Field(default_factory=lambda: {}, description="元数据")
    chunk_id: int = Field(..., description = '分块编号')
    chunk_size: int = Field(..., description = '分块大小')
    start_pos: int = Field(..., description = '分块起始位置')
    end_pos: int = Field(..., description = '分块结束位置')
    content: str = Field(..., description = '分块内容')


class DocumentMetadata(AnyDataModel):
    created: datetime.datetime = Field(default_factory=datetime.datetime.now, description="创建时间")
    name: str = Field(default_factory=lambda: "default", description="文档名称")
    topic: str = Field(default_factory=lambda: "default", description="文档主题")
    tag: str = Field(default_factory=lambda: "default", description="文档标签")
    length: int = Field(default_factory=lambda: 0, description="文档长度")


class ChunkBatcher(AnyDataModel):
    """分块批次模型"""
    batch_index: int = Field(..., description="批次索引")
    batch_size: int = Field(..., description="批次大小")
    chunks: List[Chunker] = Field(..., description="批次中的分块列表")
    
    # 继承自父文档的信息
    metadata: Optional[DocumentMetadata] = Field(default_factory=lambda: DocumentMetadata, description="文档元数据")
    
    # 批次统计信息
    actual_size: int = Field(..., description="实际批次大小")
    total_content_length: int = Field(..., description="批次内容总长度")
    start_chunk_id: int = Field(..., description="批次起始chunk_id")
    end_chunk_id: int = Field(..., description="批次结束chunk_id")
    
    def __len__(self) -> int:
        """返回批次中的分块数量"""
        return len(self.chunks)
    
    def get_content_text(self) -> str:
        """获取批次中所有分块的内容"""
        return "\n".join(chunk.content for chunk in self.chunks)
    
    def get_chunk_ids(self) -> List[int]:
        """获取批次中所有分块的ID"""
        return [chunk.chunk_id for chunk in self.chunks]
    
    def get_total_size(self) -> int:
        """获取批次中所有分块的总大小"""
        return sum(chunk.chunk_size for chunk in self.chunks)
    

class Document(AnyDataModel):
    """doc"""
    metadata: Optional[DocumentMetadata] = Field(default_factory=lambda: DocumentMetadata(), description="文档元数据")
    chunks: Optional[List[Chunker]] = Field(default_factory=list, description="分块列表")
    
    def batchIterator(self, batch_size: int = 10) -> Generator[ChunkBatcher, None, None]:
        """
        batchIterator Generator
        
        Args:
            batch_size: default 10
            
        Yields:
            ChunkBatcher: batch object
            
        Examples:
            >>> doc = Document(chunks=chunks_list)
            >>> for batch in doc.batch(batch_size=5):
            ...     print(f"batch_index {batch.batch_index}: {len(batch.chunks)}")
        """
        if not self.chunks:
            return
        
        if batch_size <= 0:
            raise ValueError("batch_size need > 0")
        
        total_chunks = len(self.chunks)
        
        for i in range(0, total_chunks, batch_size):
            # get batch
            batch_chunks = self.chunks[i:i + batch_size]
            
            # cal batch info
            batch_index = i // batch_size
            actual_size = len(batch_chunks)
            total_content_length = sum(len(chunk.content) for chunk in batch_chunks)
            start_chunk_id = batch_chunks[0].chunk_id if batch_chunks else 0
            end_chunk_id = batch_chunks[-1].chunk_id if batch_chunks else 0
            
            # create batch
            batch = ChunkBatcher(
                batch_index=batch_index,
                batch_size=batch_size,
                chunks=batch_chunks,
                metadata=self.metadata.copy(),
                actual_size=actual_size,
                total_content_length=total_content_length,
                start_chunk_id=start_chunk_id,
                end_chunk_id=end_chunk_id
            )
            
            yield batch

    def add_chunk(self, chunk: Chunker) -> None:
        """add chunk"""
        if self.chunks is None:
            self.chunks = []
        self.chunks.append(chunk)
        # update document length
        self.metadata.length = max(self.metadata.length, chunk.end_pos)
    
    def add_chunks(self, chunks: List[Chunker]) -> None:
        """batch add chunks"""
        for chunk in chunks:
            self.add_chunk(chunk)


class BaseTextChunker(abc.ABC):
    def __init__(
        self,   
    ):
        pass
    
    @abc.abstractmethod
    def invoke(self, text: str, **kwargs):
        """
        sync customize your chunker here
        """
        
    @abc.abstractmethod
    async def ainvoke(self, text: str, **kwargs):
        """
        async customize your chunker here
        """
        
        
class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"
    HASKELL = "haskell"
    ELIXIR = "elixir"
    POWERSHELL = "powershell"
    

class AnySeparators:
    
    @classmethod
    def get_separators_for_language(cls, language: Language) -> List[str]:
        if language == Language.C or language == Language.CPP:
            return [
                "\nclass ",
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.GO:
            return [
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JAVA:
            return [
                "\nclass ",
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.KOTLIN:
            return [
                "\nclass ",
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\ncase ",
                "\nelse ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JS:
            return [
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                "\nclass ",
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PHP:
            return [
                "\nfunction ",
                "\nclass ",
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PROTO:
            return [
                "\nmessage ",
                "\nservice ",
                "\nenum ",
                "\noption ",
                "\nimport ",
                "\nsyntax ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PYTHON:
            return [
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RST:
            return [
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                "\n\n.. *\n\n",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUBY:
            return [
                "\ndef ",
                "\nclass ",
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.ELIXIR:
            return [
                "\ndef ",
                "\ndefp ",
                "\ndefmodule ",
                "\ndefprotocol ",
                "\ndefmacro ",
                "\ndefmacrop ",
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\ncase ",
                "\ncond ",
                "\nwith ",
                "\nfor ",
                "\ndo ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUST:
            return [
                "\nfn ",
                "\nconst ",
                "\nlet ",
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SCALA:
            return [
                "\nclass ",
                "\nobject ",
                "\ndef ",
                "\nval ",
                "\nvar ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SWIFT:
            return [
                "\nfunc ",
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.MARKDOWN:
            return [
                "\n#{1,6} ",
                "```\n",
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.LATEX:
            return [
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                "\n\\\\begin{align}",
                "$$",
                "$",
                " ",
                "",
            ]
        elif language == Language.HTML:
            return [
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        elif language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                "\nclass ",
                "\nabstract ",
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SOL:
            return [
                "\npragma ",
                "\nusing ",
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.COBOL:
            return [
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                "\nINPUT-OUTPUT SECTION.",
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                "\n",
                " ",
                "",
            ]
        elif language == Language.LUA:
            return [
                "\nlocal ",
                "\nfunction ",
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nrepeat ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.HASKELL:
            return [
                "\nmain :: ",
                "\nmain = ",
                "\nlet ",
                "\nin ",
                "\ndo ",
                "\nwhere ",
                "\n:: ",
                "\n= ",
                "\ndata ",
                "\nnewtype ",
                "\ntype ",
                "\n:: ",
                "\nmodule ",
                "\nimport ",
                "\nqualified ",
                "\nimport qualified ",
                "\nclass ",
                "\ninstance ",
                "\ncase ",
                "\n| ",
                "\ndata ",
                "\n= {",
                "\n, ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.POWERSHELL:
            return [
                "\nfunction ",
                "\nparam ",
                "\nif ",
                "\nforeach ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\nclass ",
                "\ntry ",
                "\ncatch ",
                "\nfinally ",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language in Language._value2member_map_:
            raise ValueError(f"Language {language} is not implemented yet!")
        else:
            raise ValueError(
                f"Language {language} is not supported! "
                f"Please choose from {list(Language)}"
            )
