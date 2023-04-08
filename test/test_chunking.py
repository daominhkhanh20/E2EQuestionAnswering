import json
from e2eqavn.processor import ChunkingDocument

with open('data/UITSquad/train.json', 'r') as file:
    data = json.load(file)

chunking_document = ChunkingDocument.chunking_document_from_dict(
    data=data['data'],
    key_paragraph='paragraphs',
    key_context='context',
    max_length=256,
    overlapping_size=50
)
