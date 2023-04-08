from typing import List, Any, Dict, Tuple
import uuid
import math
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class DocumentChunk:
    def __init__(self, content_id: str, context: str, document_id: str, title: str = None):
        self.id_content = content_id
        self.context = context
        self.document_id = document_id
        self.title = title

    @classmethod
    def init_document(cls, context, document_id: str, title: str = None):
        content_id = str(uuid.uuid1())
        return cls(content_id, context, document_id, title)


class ChunkingDocument:
    def __init__(self, max_length: int, overlapping_size: int, documents: List[DocumentChunk]):
        self.max_length = max_length
        self.overlapping_size = overlapping_size
        self.documents = documents

    @classmethod
    def chunking_document_from_dict(cls,
                                    data: List[dict],
                                    key_context: str,
                                    key_paragraph: str,
                                    max_length: int = 256,
                                    overlapping_size: int = 30):
        list_documents = []
        if max_length < overlapping_size:
            raise Exception(f"max_length value must greater than overlapping size")
        cnt = 0
        size = max_length - overlapping_size
        for paragraphs in tqdm(data, total=len(data)):
            title = paragraphs['title']
            for paragraph in paragraphs[key_paragraph]:
                cnt += 1
                context = paragraph[key_context]
                list_words = context.split(" ")
                len_document = len(list_words)
                n_chunk = math.ceil(len_document / size)
                document_id = str(uuid.uuid1())
                for i in range(n_chunk):
                    temp_document = " ".join(list_words[i * size: i * size + max_length])
                    list_documents.append(DocumentChunk.init_document(temp_document, document_id, title=title))
        logger.info(f"Before chunking: {cnt} documents")
        logger.info(f"After chunking: {len(list_documents)} documents")
        return cls(max_length, overlapping_size, list_documents)
