from e2eqavn.utils.io import *

corpus = load_json_data('data/hand/edu_train.json')
hand = load_json_data('data/hand/data1.json')

list_context = [document['context'] for document in corpus]
for tmp_document in hand:
    if len(tmp_document['qas']) == 0:
        continue
    if tmp_document['context'] not in list_context:
        corpus.append(tmp_document)
        print("add 1")

write_json_file(corpus, 'data2.json')