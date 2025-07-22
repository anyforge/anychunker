import re
import numpy as np
from typing import (
    Callable, Optional, Iterable, Any,
    List, Union, Literal
)
from .base import (
    Document, DocumentMetadata, Chunker, 
    BaseTextChunker, AnySeparators, Language
)


def split_by_regex(regex: str = None):
    if regex:
        regex = re.compile(regex)
    else:
        regex = re.compile("[^,;；。\n]+[,;；。\n]?")
    return lambda text: re.findall(regex, text)


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


class AnySemanticsChunker(BaseTextChunker):
    def __init__(
        self,
        embedding_model,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        distance_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        buffer_size: Optional[int] = 1,
        breakpoint_threshold: Optional[int] = 95,
        length_function: Callable[[str], int] = len,
        strip_whitespace: bool = True,
    ):
        self.buffer_size = buffer_size
        self.breakpoint_threshold = breakpoint_threshold
        self.sentence_splitter = sentence_splitter or split_by_regex()
        self.distance_function = distance_function or cosine_similarity
        self.embedding_model = embedding_model
        self._length_function = length_function
        self._strip_whitespace = strip_whitespace
        
    def build_sentence_groups(self, text_splits: List[str]) -> List:
        sentences = [
            {
                "index": idx,
                "sentence": text,
                "combined_sentence": "",
                "combined_sentence_embedding": []
            } 
            for idx, text in enumerate(text_splits)
        ]
        for i in range(len(sentences)):
            combined_sentence = ""
            for j in range(i - self.buffer_size):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"]
            combined_sentence += sentences[i]["sentence"]
                
            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences        
    
    def calculate_groups_distances(self, sentences: List) -> List:
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = np.array(sentences[i]["combined_sentence_embedding"])
            embedding_next = np.array(sentences[i + 1]["combined_sentence_embedding"])
            similarity = self.distance_function(embedding_current, embedding_next)
            similarity = float(1 - similarity)
            distances.append(similarity)
        return distances  
    
    def build_chunks_by_distances(self, sentences: List, distances: List[float]) -> List:
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_threshold
            )

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                start_index = index + 1

            if start_index < len(sentences):
                combined_text = "".join(
                    [d["sentence"] for d in sentences[start_index:]]
                )
                chunks.append(combined_text)
        else:
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks                  
    
    def invoke(self, text: str, **kwargs):
        if not text:
            return Document()
        text_splits = self.sentence_splitter(text)

        sentences = self.build_sentence_groups(text_splits)

        # embedding output format: [[1.0,1.0], [emb], ..., [emb]]
        combined_sentence_embeddings = self.embedding_model(
            [s["combined_sentence"] for s in sentences]
        )

        for idx, embedding in enumerate(combined_sentence_embeddings):
            sentences[idx]["combined_sentence_embedding"] = embedding 
            
        distances = self.calculate_groups_distances(sentences)   

        final_chunks = self.build_chunks_by_distances(sentences,distances)
        
        document_metadata = kwargs.pop("document_metadata", {})
        document_metadata = DocumentMetadata(**document_metadata)
        document_metadata.length = self._length_function(text)
        doc_result = Document(metadata=document_metadata)
        chunk_metadata = kwargs.pop("chunk_metadata", {})

        index = 0
        previous_chunk_len = 0
        for idx,content in enumerate(final_chunks):
            if self._strip_whitespace:
                content = content.strip()
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