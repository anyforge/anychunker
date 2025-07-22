import re
from typing import (
    Callable, Optional, Iterable, Any,
    List, Union, Literal
)
from .base import (
    Document, DocumentMetadata, Chunker, 
    BaseTextChunker, AnySeparators, Language
)


class AnyTextChunker(BaseTextChunker):
    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 200,  
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        strip_whitespace: bool = True,   
    ):
        if chunk_size < chunk_overlap:
            raise ValueError(
                "Chunk size cannot be less than chunk overlap."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex
        self._strip_whitespace = strip_whitespace 
        self._separators = separators or ["\n\n", "\n", " ", ""]
        
    @classmethod
    def from_tokenizer(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        from transformers import AutoTokenizer
        chunk_size = int(kwargs.pop('chunk_size', 2048))
        chunk_overlap = int(kwargs.pop('chunk_overlap', 200))  
        separators = kwargs.pop('separators', None)
        keep_separator = kwargs.pop('keep_separator', True)
        is_separator_regex = kwargs.pop('is_separator_regex', False)
        strip_whitespace = kwargs.pop('strip_whitespace', True) 
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        def _tokenizer_length(text: str) -> int:
            return len(tokenizer.encode(text))
        return cls(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,  
            separators = separators,
            length_function = _tokenizer_length,
            keep_separator = keep_separator,
            is_separator_regex = is_separator_regex,
            strip_whitespace = strip_whitespace,              
        )
        
    @classmethod
    def from_language(cls, language: Language, **kwargs: Any):
        separators = AnySeparators.get_separators_for_language(language)
        return cls(
            separators = separators,
            **kwargs
        )
             
    def split_text_with_regex(
        self,
        text: str, 
        separator: str, 
        keep_separator: Union[bool, Literal["start", "end"]] = True
    ) -> List[str]:
        """
        使用正则表达式分割文本，支持保留分隔符
        
        Args:
            text: 要分割的文本
            separator: 分隔符（支持正则表达式）
            keep_separator: 分隔符保留方式
                - False: 不保留分隔符
                - True/"start": 将分隔符添加到下一个片段开头
                - "end": 将分隔符添加到上一个片段末尾
        
        Returns:
            分割后的文本列表（过滤掉空字符串）
        """
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = (
                    ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                    if keep_separator == "end"
                    else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
                )
                if len(_splits) % 2 == 0:
                    splits += _splits[-1:]
                splits = (
                    (splits + [_splits[-1]])
                    if keep_separator == "end"
                    else ([_splits[0]] + splits)
                )
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        if not docs:
            return None
            
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
            
        return text if text else None
        
    def merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self.join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self.join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs    
    
    def recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        递归分割文本
        
        Args:
            text: 要分割的文本
            separators: 分隔符列表
            
        Returns:
            分割后的文本块列表
        """
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self.split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self.merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self.recursive_split(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self.merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def invoke(self, text: str, **kwargs):
        if not text:
            return Document()
        document_metadata = kwargs.pop("document_metadata", {})
        document_metadata = DocumentMetadata(**document_metadata)
        document_metadata.length = self._length_function(text)
        doc_result = Document(metadata=document_metadata)
        chunk_metadata = kwargs.pop("chunk_metadata", {})
        final_chunks = self.recursive_split(text, self._separators)
        index = 0
        previous_chunk_len = 0
        for idx,content in enumerate(final_chunks):
            offset = index + previous_chunk_len - self._chunk_overlap
            index = text.find(content, max(0, offset))
            previous_chunk_len = self._length_function(content)
            current_chunk_metadata = chunk_metadata.copy()
            chunk = Chunker(
                chunk_id=idx,
                chunk_size=self._length_function(content),
                content=content,
                start_pos=index,
                end_pos=index + self._length_function(content),
                metadata=current_chunk_metadata
            )
            doc_result.add_chunk(chunk)
        
        return doc_result
        
        
    async def ainvoke(self, text: str, **kwargs):
        """
        异步版本的invoke方法
        """
        # 由于当前实现不涉及真正的异步操作，直接调用同步版本
        # 在实际应用中，可以在这里添加异步I/O操作
        return self.invoke(text, **kwargs)

