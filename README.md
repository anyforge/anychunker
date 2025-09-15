# AnyChunker

Split any text for LLM or RAG or Agent.

- text, semantics, token, markdown, code language, custom, etc.

```
    _                 ____ _                 _
   / \   _ __  _   _ / ___| |__  _   _ _ __ | | _____ _ __
  / _ \ | '_ \| | | | |   | '_ \| | | | '_ \| |/ / _ \ '__|
 / ___ \| | | | |_| | |___| | | | |_| | | | |   <  __/ |
/_/   \_\_| |_|\__, |\____|_| |_|\__,_|_| |_|_|\_\___|_|
               |___/

```

## install

```bash
pip install .

or

pip install -e .
```

## 1. recursive split text

```python
from anychunker.text import AnyTextChunker

text = """
# 1111
## 1111.22
dsdsdsds

## 1.4 dsdsdd
dajajfsdfds
###### dsdsdsd
"""
```

### by regex split

```python
## by regex split

model1 = AnyTextChunker(chunk_size = 50, chunk_overlap = 0)
model1.invoke(text)


    Document(metadata=DocumentMetadata(created=datetime.datetime(2025, 7, 22, 15, 1, 59, 827280), name='default', topic='default', tag='default', length=70), chunks=[Chunker(metadata={}, chunk_id=0, chunk_size=26, start_pos=1, end_pos=27, content='# 1111\n## 1111.22\ndsdsdsds'), Chunker(metadata={}, chunk_id=1, chunk_size=40, start_pos=29, end_pos=69, content='## 1.4 dsdsdd\ndajajfsdfds\n###### dsdsdsd')])
```

### auto batch doc

```python
for x in model1.invoke(text).batchIterator(batch_size = 1):
    print(x,'\n\n')



    ChunkBatcher(batch_index=0, batch_size=1, chunks=[Chunker(metadata={}, chunk_id=0, chunk_size=26, start_pos=1, end_pos=27, content='# 1111\n## 1111.22\ndsdsdsds')], metadata=DocumentMetadata(created=datetime.datetime(2025, 7, 22, 15, 2, 2, 861414), name='default', topic='default', tag='default', length=70), actual_size=1, total_content_length=26, start_chunk_id=0, end_chunk_id=0) 
  
  
    ChunkBatcher(batch_index=1, batch_size=1, chunks=[Chunker(metadata={}, chunk_id=1, chunk_size=40, start_pos=29, end_pos=69, content='## 1.4 dsdsdd\ndajajfsdfds\n###### dsdsdsd')], metadata=DocumentMetadata(created=datetime.datetime(2025, 7, 22, 15, 2, 2, 861414), name='default', topic='default', tag='default', length=70), actual_size=1, total_content_length=40, start_chunk_id=1, end_chunk_id=1) 
```

### by transformer tokenizer

```python
## by transformer tokenizer
model2 = AnyTextChunker.from_tokenizer("Qwen/Qwen3-8B",chunk_size = 50, chunk_overlap = 0)
model2.invoke(text)


    Document(metadata=DocumentMetadata(created=datetime.datetime(2025, 7, 22, 14, 57, 45, 431157), name='default', topic='default', tag='default', length=45), chunks=[Chunker(metadata={}, chunk_id=0, chunk_size=43, start_pos=1, end_pos=44, content='# 1111\n## 1111.22\ndsdsdsds\n\n## 1.4 dsdsdd\ndajajfsdfds\n###### dsdsdsd')])
```

### by any tokenizer

```python
import jieba

def _tokenizer_length(text: str) -> int:
    return len(jieba.lcut(text))

model1 = AnyTextChunker(chunk_size = 50, chunk_overlap = 0, length_function = _tokenizer_length,)
model1.invoke(text)

```

### by language

```python
## by language
from anychunker.base import Language

model3 = AnyTextChunker.from_language(Language.MARKDOWN,chunk_size = 50, chunk_overlap = 0)
model3.invoke(text)


    Document(metadata=DocumentMetadata(created=datetime.datetime(2025, 7, 22, 14, 59, 4, 222919), name='default', topic='default', tag='default', length=70), chunks=[Chunker(metadata={}, chunk_id=0, chunk_size=26, start_pos=1, end_pos=27, content='# 1111\n## 1111.22\ndsdsdsds'), Chunker(metadata={}, chunk_id=1, chunk_size=40, start_pos=29, end_pos=69, content='## 1.4 dsdsdd\ndajajfsdfds\n###### dsdsdsd')])
```

## 2. super markdown header split

```python
from anychunker.markdown import AnyMarkdownChunker

text = """
# 1111
## 1111.22
dsdsdsds

## 1.4 dsdsdd
dajajfsdfds
###### dsdsdsd
"""
model4 = AnyMarkdownChunker([('#','header1'),('##','Header2')])
model4.invoke(text)


   Document(metadata=DocumentMetadata(created=datetime.datetime(2025, 7, 22, 15, 2, 55, 76103), name='default', topic='default', tag='default', length=70), chunks=[Chunker(metadata={'header1': '1111', 'Header2': '1111.22'}, chunk_id=0, chunk_size=8, start_pos=19, end_pos=27, content='dsdsdsds'), Chunker(metadata={'header1': '1111', 'Header2': '1.4 dsdsdd'}, chunk_id=1, chunk_size=26, start_pos=43, end_pos=69, content='dajajfsdfds\n###### dsdsdsd')])
```

## 3. Semantics text split

```python
from anychunker.semantics import AnySemanticsChunker
from sentence_transformers import SentenceTransformer

# Load the model
model_dir = "Qwen/Qwen3-Embedding-0.6B"
model = SentenceTransformer(model_dir)

def emb_model(sentences):
    return model.encode(sentences).tolist()


model5 = AnySemanticsChunker(embedding_model = emb_model)

text = """
# 1111
## 1111.22
dsdsdsds.

## 1.4 dsdsdd
dajajfsdfds.
###### dsdsdsd
"""

model5.invoke(text)

Document(metadata=DocumentMetadata(created=datetime.datetime(2025, 7, 22, 16, 9, 12, 397166), name='default', topic='default', tag='default', length=72), chunks=[Chunker(metadata={}, chunk_id=0, chunk_size=17, start_pos=1, end_pos=18, content='# 1111\n## 1111.22'), Chunker(metadata={}, chunk_id=1, chunk_size=51, start_pos=-1, end_pos=50, content='dsdsdsds.\n## 1.4 dsdsdd\ndajajfsdfds.\n###### dsdsdsd')])
```

```python
# see all functions

docs = model5.invoke(text)

dir(docs)
```
