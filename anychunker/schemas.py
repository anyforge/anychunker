import json
import datetime
from abc import ABC
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Generator, Literal, Any , Any, Dict, Set, List


class AnyDataModel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True,
        arbitrary_types_allowed=True,
        # V2 中 json_encoders 的用法保持不变
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat(),
            datetime.time: lambda v: v.isoformat(),
        }
    )
        
    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'

    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"字段 '{key}' 不存在")
    
    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)
    
    def __delitem__(self, key: str):
        if hasattr(self, key):
            delattr(self, key)
        else:
            raise KeyError(f"字段 '{key}' 不存在")
    
    def __contains__(self, key: str):
        return hasattr(self, key)
    
    def __len__(self):
        return len(self.model_fields_set) + len(self.model_extra or {})
    
    def __iter__(self):
        return iter(self.keys())

    def __eq__(self, other: Any):
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
        return not self.__eq__(other)
    
    def __hash__(self):
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
        try:
            hash(obj)
            return True
        except TypeError:
            return False
    
    def equals(self, other, 
               ignore_fields: Set[str] = None,
               ignore_created: bool = True):
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
        data = self.model_dump()
        return data.keys()
    
    def values(self):
        data = self.model_dump()
        return data.values()
    
    def items(self):
        data = self.model_dump()
        return data.items()
    
    def get(self, key: str, default: Any = None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def pop(self, key: str, default: Any = None):
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
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default
    
    def clear(self):
        all_keys = list(self.keys())
        for key in all_keys:
            if key not in self.keys():
                try:
                    del self[key]
                except (KeyError, AttributeError):
                    pass
        
        for field_name, field_info in self.model_fields().items():
            if field_info.default is not None:
                self[field_name] = field_info.default
            elif field_info.default_factory is not None:
                self[field_name] = field_info.default_factory()
                
    # 增强的复制方法
    def copy(self, deep: bool = True, update: Dict[str, Any] = None):
        return self.model_copy(deep=deep, update=update)

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
    

class Documents(AnyDataModel):
    """doc"""
    metadata: Optional[DocumentMetadata] | dict = Field(default_factory=lambda: DocumentMetadata(), description="文档元数据")
    chunks: Optional[List[Chunker]] | list = Field(default_factory=list, description="分块列表")
    
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