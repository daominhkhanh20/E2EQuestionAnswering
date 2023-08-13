from e2eqavn.utils.io import *

corpus = load_json_data('data/hand/data1.json')
new_corpus = []
for document in corpus:
    new_corpus.append({
        "context": document['text'],
        "qas": document['qas']
    })

write_json_file(new_corpus, 'data/hand/data1.json')
# hand = load_json_data('data/hand/edu_test.json')
#
# new_corpus = []
# for document in corpus:
#     flag_exist = False
#     for tmp_document in hand:
#         if document['text'] == tmp_document['context']:
#             new_corpus.append({
#                 "text": document['text'],
#                 "id": document['id'],
#                 "qas": tmp_document['qas']
#             })
#             flag_exist = True
#             break
#     if not flag_exist:
#         new_corpus.append({
#             "text": document['text'],
#             "id": document['id'],
#             "qas": [] if len(document['qas']) == 0 or document['qas'][0]["question"] == ""  else document['qas']
#         })
# write_json_file(new_corpus, 'data1.json')