from pymongo import MongoClient
import json
from e2eqavn.utils.io import *

client = MongoClient(
    "mongodb+srv://dataintergration:nhom10@cluster0.hqw7c.mongodb.net/test")

database = client['wikipedia']
wiki_collections = database['Process_Education2']

data = load_json_data('data/hand/edu_train.json')
corpus = []
for document in wiki_collections.find({}):
    corpus.append({
        "text": document['text'],
        "id": str(document['_id']),
        "qas": [
            {
                "question": "",
                "answers": [
                    {
                        "text": ""
                    }
                ]
            }
        ]
    })
    if data[0]['context'] == document['text']:
        print("exist")

write_json_file(corpus, path_file='data.json')