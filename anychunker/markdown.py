import re
from typing import (
    Callable, Optional, Iterable, Any,
    List, Tuple, Union, Literal
)
from .text import AnyTextChunker
from .base import BaseTextChunker,AnySeparators, Language
from .schemas import (
    Documents, DocumentMetadata, Chunker
)


class AnyMarkdownChunker(BaseTextChunker):
    DEFAULT_HEADER_KEYS = {
        "#": "Header-1",
        "##": "Header-2",
        "###": "Header-3",
    }
    def __init__(
        self,
        headers_to_split_on: Union[List[Tuple[str, str]], None] = None,
        strip_headers: bool = True, 
        return_each_line: bool = False,
        length_function: Callable[[str], int] = len
    ):
        if headers_to_split_on:
            splittable_headers = dict(headers_to_split_on)
        else:
            splittable_headers = self.DEFAULT_HEADER_KEYS
        self.headers_to_split_on = sorted(
            splittable_headers.items(), key=lambda split: len(split[0]), reverse=True
        )
        self.headers_to_split_on_dict = dict([(y,x) for x,y in self.headers_to_split_on])
        self.strip_headers = strip_headers
        self.return_each_line = return_each_line
        self._length_function = length_function
    
    @classmethod
    def from_markdown_language(**kwargs):
        return AnyTextChunker.from_language(Language.MARKDOWN, **kwargs)
    
    def recursive_split(self, text: str) -> List[dict]:
        lines = text.split("\n")
        lines_with_metadata = []
        current_content = []
        current_metadata = {}
        header_stack = []
        initial_metadata = {}
        in_code_block = False
        opening_fence = ""
        for line in lines:
            # stripped_line = line.strip()
            stripped_line = line
            stripped_line = "".join(filter(str.isprintable, stripped_line))
            if not in_code_block:
                # Exclude inline code spans
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                elif stripped_line.startswith("~~~"):
                    in_code_block = True
                    opening_fence = "~~~"
            else:
                if stripped_line.startswith(opening_fence):
                    in_code_block = False
                    opening_fence = ""

            if in_code_block:
                current_content.append(stripped_line)
                continue

            # Check each line against each of the header types (e.g., #, ##)
            for sep, name in self.headers_to_split_on:
                # Check if line starts with a header that we intend to split on
                if stripped_line.startswith(sep) and (
                    # Header with no text OR header is followed by space
                    # Both are valid conditions that sep is being used a header
                    len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                ):
                    # Ensure we are tracking the header as metadata
                    if name is not None:
                        # Get the current header level
                        current_header_level = sep.count("#")

                        # Pop out headers of lower or same level from the stack
                        while (
                            header_stack
                            and header_stack[-1]["level"] >= current_header_level
                        ):
                            # We have encountered a new header
                            # at the same or higher level
                            popped_header = header_stack.pop()
                            # Clear the metadata for the
                            # popped header in initial_metadata
                            if popped_header["name"] in initial_metadata:
                                initial_metadata.pop(popped_header["name"])

                        # Push the current header to the stack
                        header = {
                            "level": current_header_level,
                            "name": name,
                            "data": stripped_line[len(sep) :].strip(),
                        }
                        header_stack.append(header)
                        # Update initial_metadata with the current header
                        initial_metadata[name] = header["data"]

                    # Add the previous line to the lines_with_metadata
                    # only if current_content is not empty
                    if current_content:
                        lines_with_metadata.append(
                            {
                                "content": "\n".join(current_content),
                                "metadata": current_metadata.copy(),
                            }
                        )
                        current_content.clear()

                    if not self.strip_headers:
                        current_content.append(stripped_line)

                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append(
                        {
                            "content": "\n".join(current_content),
                            "metadata": current_metadata.copy(),
                        }
                    )
                    current_content.clear()

            current_metadata = initial_metadata.copy()

        if current_content:
            lines_with_metadata.append(
                {
                    "content": "\n".join(current_content),
                    "metadata": current_metadata,
                }
            )
        if self.return_each_line:
            final_chunks = []
            for chunk in lines_with_metadata:
                final_chunks.append({
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                })
        else:
            final_chunks = self.aggregate_lines_to_chunks(lines_with_metadata)
        return final_chunks  
    
    def aggregate_lines_to_chunks(self, lines: list) -> list[dict]:
        """Combine lines with common metadata into chunks.

        Args:
            lines: Line of text / associated header metadata
        """
        aggregated_chunks = []

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                # If the last line in the aggregated list
                # has the same metadata as the current line,
                # append the current content to the last lines's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
            elif (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] != line["metadata"]
                # may be issues if other metadata is present
                and len(aggregated_chunks[-1]["metadata"]) < len(line["metadata"])
                and aggregated_chunks[-1]["content"].split("\n")[-1][0] == "#"
                and not self.strip_headers
            ):
                # If the last line in the aggregated list
                # has different metadata as the current line,
                # and has shallower header level than the current line,
                # and the last line is a header,
                # and we are not stripping headers,
                # append the current content to the last line's content
                aggregated_chunks[-1]["content"] += "  \n" + line["content"]
                # and update the last line's metadata
                aggregated_chunks[-1]["metadata"] = line["metadata"]
            else:
                # Otherwise, append the current line to the aggregated list
                aggregated_chunks.append(line)
                
        final_chunks = []
        for chunk in aggregated_chunks:
            final_chunks.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            })

        return final_chunks
    def invoke(self, text: str, **kwargs):
        if not text:
            return Documents()
        document_metadata = kwargs.pop("document_metadata", {})
        document_metadata = DocumentMetadata(**document_metadata)
        document_metadata.length = self._length_function(text)
        doc_result = Documents(metadata=document_metadata)
        chunk_metadata = kwargs.pop("chunk_metadata", {})
        final_chunks = self.recursive_split(text)
        index = 0
        previous_chunk_len = 0
        for idx,item in enumerate(final_chunks):
            content = item['content']
            current_chunk_metadata = chunk_metadata.copy()
            current_chunk_metadata.update(item['metadata'])
            offset = index + previous_chunk_len
            index = text.find(content, max(0, offset))
            previous_chunk_len = self._length_function(content)
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