import json 

def load_json(path_file):
    with open(path_file, 'r') as file:
        data = json.load(file)
    return data 

def write_json(data, path_file):
    with open(path_file, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

data = load_json('edu.json')
ratio = 0.8
n_document = len(data)
n_document_train = int(ratio * n_document)
data_train = data[:n_document_train]
data_test = data[n_document_train:]
print(f"N document train: {len(data_train)} -- N question: {sum([len(doc['qas']) for doc in data_train])}")
print(f"N document test: {len(data_test)} -- N question test: {sum([len(doc['qas']) for doc in data_test])}")

write_json(data_train, 'edu_train.json')
write_json(data_test, 'edu_test.json')
