import re
from typing import (
    Callable, Optional, Iterable, Any,
    List, Union, Literal, Dict, Set, Tuple
)
from .text import AnyTextChunker
from .markdown import AnyMarkdownChunker
from .base import BaseTextChunker,AnySeparators, Language
from .schemas import (
    Documents, DocumentMetadata, Chunker
)


class AnyMarkdownBlockChunker(BaseTextChunker):
    """
    智能Markdown文本切分器，能够识别并保护特殊块结构
    """
    DEFAULT_HEADER_KEYS = {
        "#": "Header-1",
        "##": "Header-2",
        "###": "Header-3",
    }
    def __init__(
        self,
        chunk_size: int = 2048,
        chunk_overlap: int = 200,  
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        strip_whitespace: bool = True,    
        block_patterns: Dict[str, str] = None,
        headers_to_split_on: Union[List[Tuple[str, str]], None] = None,
        strip_headers: bool = True, 
        return_each_line: bool = False,
        title_dash: str = " » ",
        heading_dash: str = " » "
    ):
        """
        初始化切分器
        
        Args:
            chunk_size: 每个chunk的目标大小
            chunk_overlap: chunk之间的重叠大小
            separators: 文本切分的分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", " ", ""]
        self.length_function = length_function
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self.strip_whitespace = strip_whitespace
        
        self.text_splitter = AnyTextChunker(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,  
            separators = self.separators,
            length_function = self.length_function,
            keep_separator = self.keep_separator,
            is_separator_regex = self.is_separator_regex,
            strip_whitespace = self.strip_whitespace               
        )

        if headers_to_split_on:
            splittable_headers = dict(headers_to_split_on)
        else:
            splittable_headers = self.DEFAULT_HEADER_KEYS
        self.headers_to_split_on = sorted(
            splittable_headers.items(), key=lambda split: len(split[0]), reverse=True
        )
        self.headers_to_split_on = [(x,y) for x,y in self.headers_to_split_on]
        self.strip_headers = strip_headers
        self.return_each_line = return_each_line
        
        self.markdown_splitter = AnyMarkdownChunker(
            headers_to_split_on = self.headers_to_split_on,
            strip_headers = self.strip_headers, 
            return_each_line = self.return_each_line,
            length_function = self.length_function
        )
        
        # 定义需要保护的块模式（字典格式：类型 -> 正则表达式）
        self.block_patterns = block_patterns or {
            'front_matter': r'^---\n.*?\n---\n',
            'code_block': r'```[\s\S]*?```',
            'indented_code': r'(?:^|\n)((?:(?:    |\t).*\n)+)',
            'math_block': r'\$\$[\s\S]*?\$\$',
            'html_table': r'<table[\s\S]*?</table>',
            'html_div': r'<div[\s\S]*?</div>',
            'html_details': r'<details[\s\S]*?</details>',
            'html_section': r'<section[\s\S]*?</section>',
            'html_article': r'<article[\s\S]*?</article>',
            'markdown_table': r'(?:^|\n)(\|.+\|[\n\r]+\|[-:\s|]+\|(?:[\n\r]+\|.+\|)*)',
            'extended_fence': r'```(?:mermaid|plantuml|admonition|note|warning|tip)[\s\S]*?```',
            'blockquote': r'(?:^|\n)((?:>.*(?:\n|$))+)',
            'ordered_list': r'(?:^|\n)((?:[ \t]*\d+\.[ \t]+.*(?:\n|$)(?:[ \t]+.*(?:\n|$))*)+)',
            'unordered_list': r'(?:^|\n)((?:[ \t]*[-*+][ \t]+.*(?:\n|$)(?:[ \t]+.*(?:\n|$))*)+)',
            'task_list': r'(?:^|\n)((?:[ \t]*-[ \t]+\[[ xX]\][ \t]+.*(?:\n|$))+)',
            'footnote': r'(?:^|\n)(\[\^[^\]]+\]:[ \t]+.*(?:\n(?:[ \t]+.*)?)*)',
        }
        self.title_dash = title_dash or " » "
        self.heading_dash = heading_dash or " » "
    
    def _extract_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文本中的所有特殊块
        
        Returns:
            包含块信息的列表，每个元素为 {start, end, content, type}
        """
        blocks = []
        
        for block_type, pattern in self.block_patterns.items():
            try:
                for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
                    blocks.append({
                        'start': match.start(),
                        'end': match.end(),
                        'content': match.group(0),
                        'type': block_type,
                    })
            except re.error as e:
                print(f"警告: 模式 '{pattern}' 出错: {e}")
                continue
        
        # 按起始位置排序并处理重叠（保留最外层的块）
        blocks.sort(key=lambda x: (x['start'], -x['end']))
        
        # 去除重叠的块
        filtered_blocks = []
        last_end = -1
        for block in blocks:
            if block['start'] >= last_end:
                filtered_blocks.append(block)
                last_end = block['end']
        
        return filtered_blocks
    
    def _split_with_blocks(self, text: str, blocks: List[Dict[str, Any]]) -> List[str]:
        """
        智能切分：将文本分成块和非块部分，分别处理
        """
        if not blocks:
            res = self.text_splitter.invoke(text)
            res = [x.content for x in res.chunks]
            return res
        
        # 构建文本段列表：[{'type': 'text'/'block', 'content': '...'}]
        segments = []
        last_pos = 0
        
        for block in blocks:
            # 添加块之前的文本
            if block['start'] > last_pos:
                text_content = text[last_pos:block['start']]
                if text_content.strip():
                    segments.append({
                        'type': 'text',
                        'content': text_content
                    })
            
            # 添加块本身
            segments.append({
                'type': 'block',
                'content': block['content']
            })
            
            last_pos = block['end']
        
        # 添加最后剩余的文本
        if last_pos < len(text):
            text_content = text[last_pos:]
            if text_content.strip():
                segments.append({
                    'type': 'text',
                    'content': text_content
                })
        
        # 处理每个段落
        processed_chunks = []
        
        for segment in segments:
            if segment['type'] == 'block':
                # 块内容保持完整
                processed_chunks.append(segment['content'])
            else:
                # 文本内容进行切分
                text_chunks = self.text_splitter.invoke(segment['content'])
                text_chunks = [x.content for x in text_chunks.chunks]
                processed_chunks.extend(text_chunks)
        
        return processed_chunks
    
    def split_content(self, content: str) -> List[str]:
        """
        切分单个content内容
        
        Args:
            content: 要切分的文本内容
            
        Returns:
            切分后的文本块列表
        """
        if not content or not content.strip():
            return [content] if content else []
        
        # 1. 提取所有特殊块
        blocks = self._extract_blocks(content)
        
        # 2. 如果整个内容就是一个块，直接返回
        if len(blocks) == 1 and blocks[0]['start'] == 0 and blocks[0]['end'] == len(content):
            return [content]
        
        # 3. 智能切分：分别处理块和文本
        chunks = self._split_with_blocks(content, blocks)
        
        # 4. 过滤空白chunk
        chunks = [c for c in chunks if c.strip()]
        
        return chunks
    
    def _get_all_ancestors(self, id_value: int, direct_parents: Dict[int, List[int]]) -> List[int]:
        """
        递归获取所有祖先节点（使用BFS，按层级从近到远）
        
        Args:
            id_value: 当前节点的id
            direct_parents: 直接父节点映射 {id: [parent_id1, parent_id2, ...]}
        
        Returns:
            所有祖先节点的id列表（按从近到远排序）
        """
        ancestors = []
        visited = set()
        
        # BFS: 按层级遍历
        current_level = direct_parents.get(id_value, [])[:]  # 复制列表
        
        while current_level:
            next_level = []
            for parent_id in current_level:
                if parent_id not in visited:
                    visited.add(parent_id)
                    ancestors.append(parent_id)
                    # 获取父节点的父节点
                    grandparents = direct_parents.get(parent_id, [])
                    next_level.extend(grandparents)
            current_level = next_level
        
        return ancestors
    
    def _get_all_descendants(self, id_value: int, direct_children: Dict[int, List[int]]) -> List[int]:
        """
        递归获取所有后代节点（使用BFS，按层级从近到远）
        
        Args:
            id_value: 当前节点的id
            direct_children: 直接子节点映射 {id: [child_id1, child_id2, ...]}
        
        Returns:
            所有后代节点的id列表（按从近到远排序）
        """
        descendants = []
        visited = set()
        
        # BFS: 按层级遍历
        current_level = direct_children.get(id_value, [])[:]  # 复制列表
        
        while current_level:
            next_level = []
            for child_id in current_level:
                if child_id not in visited:
                    visited.add(child_id)
                    descendants.append(child_id)
                    # 获取子节点的子节点
                    grandchildren = direct_children.get(child_id, [])
                    next_level.extend(grandchildren)
            current_level = next_level
        
        return descendants
    
    def _build_block_relationships(self, blocks: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[int]]]:
        """
        构建block之间的父子关系
        
        Returns:
            {block_id: {'block_parent_id': [...], 'block_child_id': [...]}}
        """
        block_ids = sorted([b['block_id'] for b in blocks])
        
        # 建立直接父子关系
        direct_parents = {}
        direct_children = {}
        
        for i, block_id in enumerate(block_ids):
            # 前一个block是父节点
            if i > 0:
                direct_parents[block_id] = [block_ids[i-1]]
            else:
                direct_parents[block_id] = []
            
            # 后一个block是子节点
            if i < len(block_ids) - 1:
                direct_children[block_id] = [block_ids[i+1]]
            else:
                direct_children[block_id] = []
        
        # 递归获取所有祖先和后代
        block_relationships = {}
        for block_id in block_ids:
            block_relationships[block_id] = {
                'block_parent_id': self._get_all_ancestors(block_id, direct_parents),
                'block_child_id': self._get_all_descendants(block_id, direct_children)
            }
        
        return block_relationships
    
    def _build_chunk_relationships(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        构建chunk之间的父子关系（递归，包含所有祖先和后代）
        
        策略：
        - 相邻的chunk之间有直接父子关系
        - chunk_parent_id 包含所有祖先（父亲、爷爷、曾爷爷...）
        - chunk_child_id 包含所有后代（儿子、孙子、曾孙...）
        """
        result = []
        
        # 按block_id分组
        block_groups = {}
        for chunk in chunks:
            block_id = chunk['block_id']
            if block_id not in block_groups:
                block_groups[block_id] = []
            block_groups[block_id].append(chunk)
        
        # 为每个block内的chunks建立父子关系
        for block_id, block_chunks in block_groups.items():
            # 按chunk_id排序
            block_chunks.sort(key=lambda x: x['chunk_id'])
            
            # 首先建立直接父子关系映射
            direct_parents = {}  # {chunk_id: [parent_chunk_id]}
            direct_children = {}  # {chunk_id: [child_chunk_id]}
            
            for i, chunk in enumerate(block_chunks):
                chunk_id = chunk['chunk_id']
                
                # 记录直接父节点
                if i > 0:
                    parent_id = block_chunks[i-1]['chunk_id']
                    direct_parents[chunk_id] = [parent_id]
                else:
                    direct_parents[chunk_id] = []
                
                # 记录直接子节点
                if i < len(block_chunks) - 1:
                    child_id = block_chunks[i+1]['chunk_id']
                    direct_children[chunk_id] = [child_id]
                else:
                    direct_children[chunk_id] = []
            
            # 然后递归获取所有祖先和后代
            for chunk in block_chunks:
                chunk_copy = chunk.copy()
                chunk_id = chunk['chunk_id']
                
                # 获取所有祖先
                chunk_copy['chunk_parent_id'] = self._get_all_ancestors(chunk_id, direct_parents)
                
                # 获取所有后代
                chunk_copy['chunk_child_id'] = self._get_all_descendants(chunk_id, direct_children)
                
                result.append(chunk_copy)
        
        return result
    
    def split_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对整个blocks列表进行处理，切分每个block的content
        
        Args:
            blocks: 包含block_id, Title, Headings, Content的字典列表
            
        Returns:
            切分后的chunks列表，每个chunk包含:
            - block_id, chunk_id
            - block_parent_id, block_child_id (block级别的祖先后代)
            - chunk_parent_id, chunk_child_id (chunk级别的祖先后代)
            - metadata (Title, Headings, Content)
        """
        # 1. 构建block级别的父子关系
        block_relationships = self._build_block_relationships(blocks)
        
        # 2. 切分每个block的content
        temp_chunks = []
        
        for block in blocks:
            block_id = block.get('block_id')
            content = block.get('Content', '')
            
            if not content or not content.strip():
                # 没有内容，保留原block但添加chunk_id
                new_chunk = {'block_id': block_id, 'chunk_id': 0}
                if 'Title' in block:
                    new_chunk['Title'] = block['Title']
                if 'Headings' in block:
                    new_chunk['Headings'] = block['Headings']
                if content:
                    new_chunk['Content'] = content
                temp_chunks.append(new_chunk)
                continue
            
            # 切分content
            split_contents = self.split_content(content)
            
            # 为每个切分后的内容创建新chunk
            for chunk_id, split_content in enumerate(split_contents):
                new_chunk = {
                    'block_id': block_id,
                    'chunk_id': chunk_id
                }
                
                # 保留Title和Headings
                if 'Title' in block:
                    new_chunk['Title'] = block['Title']
                if 'Headings' in block:
                    new_chunk['Headings'] = block['Headings']
                
                # 添加切分后的内容
                new_chunk['Content'] = split_content
                
                temp_chunks.append(new_chunk)
        
        # 3. 构建chunk级别的父子关系
        result_chunks = self._build_chunk_relationships(temp_chunks)
        
        # 4. 为每个chunk添加block级别的父子关系
        for chunk in result_chunks:
            block_id = chunk['block_id']
            chunk['block_parent_id'] = block_relationships[block_id]['block_parent_id']
            chunk['block_child_id'] = block_relationships[block_id]['block_child_id']
        
        return result_chunks
    
    def invoke_markdown(self,text: str, **kwargs):
        md_header_splits = self.markdown_splitter.invoke(text)
        chunks = []
        for idx,line in enumerate(md_header_splits.chunks):
            item = {}
            # 处理title和header
            headers = [[x,y] for x,y in line.metadata.items()]
            headers = list(sorted(headers, key = lambda x: x[0]))
            titles = []
            headings = []
            for hd in headers:
                if hd[0] == 'Header-1':
                    titles.append(hd[-1])
                else:
                    headings.append(hd[-1])
            if titles:
                # item["Title"] = f"{' » '.join(titles)}"
                item["Title"] = {
                    "name": f"{self.title_dash.join(titles)}",
                    "value": titles
                }
            if headings:
                # item["Headings"] = f"{' » '.join(headings)}"
                item["Headings"] = {
                    "name": f"{self.heading_dash.join(headings)}",
                    "value": headings
                }
            # 处理内容
            content = line.content
            if content:
                item["Content"] = content
            if item:
                newitem = {"block_id": idx}
                newitem.update(item)
                chunks.append(newitem)     
        return chunks   

    def invoke(self, text: str, **kwargs):
        if not text:
            return Documents()
        document_metadata = kwargs.pop("document_metadata", {})
        document_metadata = DocumentMetadata(**document_metadata)
        document_metadata.length = self.length_function(text)
        doc_result = Documents(metadata=document_metadata)
        markdown_res = self.invoke_markdown(text)
        blocks_res = self.split_blocks(markdown_res)
        index = 0
        previous_chunk_len = 0
        for idx,item in enumerate(blocks_res):
            offset = index + previous_chunk_len - self.chunk_overlap
            index = text.find(item['Content'], max(0, offset))
            previous_chunk_len = self.length_function(item['Content'])
            current_chunk_metadata = {
                "block_id": item['block_id'],
                "chunk_id": item['chunk_id'],
                "title": item.get("Title", {}),
                "headings": item.get("Headings", {}),
                "chunk_parent_id": item.get("chunk_parent_id", []),
                "chunk_child_id": item.get("chunk_child_id", []),
                "block_parent_id": item.get("block_parent_id", []),
                "block_child_id": item.get("block_child_id", []),
                "chunk_size": self.length_function(item['Content']),
                "chunk_overlap": self.chunk_overlap,
                "start_pos": index,
                "end_pos": index + self.length_function(item['Content']),
            }

            chunk = Chunker(
                chunk_id=current_chunk_metadata['chunk_id'],
                chunk_size=self.length_function(item['Content']),
                content=item['Content'],
                start_pos=current_chunk_metadata['start_pos'],
                end_pos=current_chunk_metadata['end_pos'],
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

# 使用示例
if __name__ == "__main__":
    markdown_document = """
    dasdasdasda
    # title1

        ## Bar
    # title1-1

        ## Bar-1
    Hi this is Jim

    ### Boo 

    Hi this is Lance 

    # title2

    ## Baz
    ```python
    # 测试
    def test():
        # 打印
        print('ok')
    ```

    |1|2|
    |---|---|
    |a1|b1|
    |a2|b2|

    Hi this is Molly
        # title3
    ## title3-header1
    dsdsd
    <table>
        <th>1111
        </th>
    </table>
    """
    from anychunker import AnyMarkdownBlockChunker
    
    chunker = AnyMarkdownBlockChunker(
        chunk_size = 100,
        chunk_overlap = 5,
    )
    
    res = chunker.invoke(markdown_document)
    res