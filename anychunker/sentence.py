import re
from typing import Callable,List, Optional
from .base import (
    Document, DocumentMetadata, Chunker, 
    BaseTextChunker, AnySeparators, Language
)


class AnySentenceChunker(BaseTextChunker):
    """中英文兼容的句子切分器 - 改进版"""
    
    def __init__(
        self, 
        chunk_size: int = 128,
        length_function: Callable[[str], int] = len
    ):
        self.chunk_size = chunk_size
        self._length_function = length_function
    
    def enhanced_rule_split(self, text: str) -> List[str]:
        """
        增强的基于规则的句子切分
        处理复杂情况：缩写、数字、引号、章节号、列表等
        """
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        protected_text = re.sub(r'\n+', '\n', text)
        
        sentences = []
            
        # 1. 保护常见英文缩写
        abbreviations = [
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Inc.', 'Ltd.', 'Co.', 'Corp.', 
            'etc.', 'vs.', 'i.e.', 'e.g.', 'cf.', 'et al.', 'Ph.D.', 'M.D.', 
            'B.A.', 'M.A.', 'LLC.', 'U.S.A.', 'U.K.', 'U.S.', 'No.', 'Fig.', 
            'Vol.', 'Ch.', 'Sec.', 'Art.', 'Dept.', 'Univ.', 'Jan.', 'Feb.', 
            'Mar.', 'Apr.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.'
        ]
        for i, abbr in enumerate(abbreviations):
            protected_text = protected_text.replace(abbr, f'__ABBR_{i}__')
        
        # 2. 保护章节号格式 (1., 1.1, 1.1.1, 1.1.1.1 等)
        # 匹配模式：数字.数字.数字... 后面跟空格或直接跟文字
        section_pattern = r'\b(\d+(?:\.\d+)*\.?)(?=\s|[A-Za-z\u4e00-\u9fff])'
        section_matches = list(re.finditer(section_pattern, protected_text))
        for i, match in enumerate(reversed(section_matches)):  # 从后往前替换避免位置错乱
            section_num = match.group(1)
            start, end = match.span()
            protected_text = protected_text[:start] + f'__SECTION_{len(section_matches)-1-i}__' + protected_text[end:]
        
        # 存储章节号以便还原
        section_replacements = {}
        for i, match in enumerate(section_matches):
            section_replacements[f'__SECTION_{i}__'] = match.group(1)
        
        # 3. 保护数字列表 (1. 2. 3. 或 1) 2) 3))
        # 匹配行首或前面有空白的数字列表
        number_list_pattern = r'(?:^|\s)(\d+[.)])\s'
        number_list_matches = list(re.finditer(number_list_pattern, protected_text))
        for i, match in enumerate(reversed(number_list_matches)):
            list_marker = match.group(1)
            start, end = match.span(1)  # 只替换数字部分
            protected_text = protected_text[:start] + f'__NUMLIST_{len(number_list_matches)-1-i}__' + protected_text[end:]
        
        # 存储数字列表标记
        number_list_replacements = {}
        for i, match in enumerate(number_list_matches):
            number_list_replacements[f'__NUMLIST_{i}__'] = match.group(1)
        
        # 4. 保护字母列表 (a. b. c. 或 a) b) c) 或 A. B. C.)
        alpha_list_pattern = r'(?:^|\s)([a-zA-Z][.)])\s'
        alpha_list_matches = list(re.finditer(alpha_list_pattern, protected_text))
        for i, match in enumerate(reversed(alpha_list_matches)):
            list_marker = match.group(1)
            start, end = match.span(1)
            protected_text = protected_text[:start] + f'__ALPHALIST_{len(alpha_list_matches)-1-i}__' + protected_text[end:]
        
        # 存储字母列表标记
        alpha_list_replacements = {}
        for i, match in enumerate(alpha_list_matches):
            alpha_list_replacements[f'__ALPHALIST_{i}__'] = match.group(1)
        
        # 5. 保护罗马数字列表 (i. ii. iii. iv. 或 I. II. III. IV.)
        roman_pattern = r'(?:^|\s)([ivxlcdmIVXLCDM]+[.)])\s'
        roman_list_matches = list(re.finditer(roman_pattern, protected_text))
        for i, match in enumerate(reversed(roman_list_matches)):
            list_marker = match.group(1)
            start, end = match.span(1)
            protected_text = protected_text[:start] + f'__ROMANLIST_{len(roman_list_matches)-1-i}__' + protected_text[end:]
        
        # 存储罗马数字列表标记
        roman_list_replacements = {}
        for i, match in enumerate(roman_list_matches):
            roman_list_replacements[f'__ROMANLIST_{i}__'] = match.group(1)
        
        # 6. 保护普通数字中的小数点
        protected_text = re.sub(r'(\d+)\.(\d+)', r'\1__DOT__\2', protected_text)
        
        # 7. 保护URL和邮箱
        protected_text = re.sub(r'https?://[^\s]+', 
                                lambda m: m.group().replace('.', '__URL_DOT__'), protected_text)
        protected_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                                lambda m: m.group().replace('.', '__EMAIL_DOT__'), protected_text)
        
        # 8. 保护时间格式 (如 3.30, 12.45)
        time_pattern = r'\b(\d{1,2})\.(\d{2})\b'
        protected_text = re.sub(time_pattern, r'\1__TIME_DOT__\2', protected_text)
        
        # 9. 保护版本号 (如 v1.0, version 2.1.3)
        version_pattern = r'\b(v\d+(?:\.\d+)*)\b'
        protected_text = re.sub(version_pattern, 
                                lambda m: m.group().replace('.', '__VERSION_DOT__'), protected_text)
        
        # 10. 保护货币和百分比
        currency_pattern = r'\$\d+(?:\.\d+)?'
        protected_text = re.sub(currency_pattern, 
                                lambda m: m.group().replace('.', '__CURRENCY_DOT__'), protected_text)
        
        # 句子分割模式
        # 中文标点
        chinese_pattern = r'([。！？；])\s*'
        # 英文标点（确保后面跟空格+大写字母或结尾）
        english_pattern = r'([.!?;])(?=\s+[A-Z\u4e00-\u9fff]|\s*$)'
        
        # 组合模式
        combined_pattern = f'({chinese_pattern}|{english_pattern})'
        
        parts = re.split(combined_pattern, protected_text)
        
        current_sentence = ""
        for part in parts:
            if not part:
                continue
            
            if re.match(r'[。！？；.!?;]', part):
                current_sentence += part
                
                # 还原所有保护的内容
                restored = self._restore_protected_content(
                    current_sentence, 
                    abbreviations, 
                    section_replacements,
                    number_list_replacements,
                    alpha_list_replacements,
                    roman_list_replacements
                )
                
                if restored.strip():
                    sentences.append(restored.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # 处理最后一个句子
        if current_sentence.strip():
            restored = self._restore_protected_content(
                current_sentence, 
                abbreviations, 
                section_replacements,
                number_list_replacements,
                alpha_list_replacements,
                roman_list_replacements
            )
            sentences.append(restored.strip())
        
        return [s for s in sentences if len(s.strip()) > 1]
    
    def _restore_protected_content(self, text: str, abbreviations: List[str], 
                                 section_replacements: dict, 
                                 number_list_replacements: dict,
                                 alpha_list_replacements: dict,
                                 roman_list_replacements: dict) -> str:
        """还原所有被保护的内容"""
        restored = text
        
        # 还原缩写
        for i, abbr in enumerate(abbreviations):
            restored = restored.replace(f'__ABBR_{i}__', abbr)
        
        # 还原章节号
        for placeholder, original in section_replacements.items():
            restored = restored.replace(placeholder, original)
        
        # 还原数字列表
        for placeholder, original in number_list_replacements.items():
            restored = restored.replace(placeholder, original)
        
        # 还原字母列表
        for placeholder, original in alpha_list_replacements.items():
            restored = restored.replace(placeholder, original)
        
        # 还原罗马数字列表
        for placeholder, original in roman_list_replacements.items():
            restored = restored.replace(placeholder, original)
        
        # 还原其他格式
        restored = restored.replace('__DOT__', '.')
        restored = restored.replace('__URL_DOT__', '.')
        restored = restored.replace('__EMAIL_DOT__', '.')
        restored = restored.replace('__TIME_DOT__', '.')
        restored = restored.replace('__VERSION_DOT__', '.')
        restored = restored.replace('__CURRENCY_DOT__', '.')
        return restored
    
    def invoke(self, text: str, **kwargs):
        if not text:
            return Document()
        document_metadata = kwargs.pop("document_metadata", {})
        document_metadata = DocumentMetadata(**document_metadata)
        document_metadata.length = self._length_function(text)
        doc_result = Document(metadata=document_metadata)
        chunk_metadata = kwargs.pop("chunk_metadata", {})
        final_chunks = self.enhanced_rule_split(text)
        index = 0
        previous_chunk_len = 0
        for idx,content in enumerate(final_chunks):
            offset = index + previous_chunk_len
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
