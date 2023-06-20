# 1. Introduction

- <b>e2eqavn </b> is a 'simple' and 'easy to use' library that provides end to end pipeline for Question answering task. 

- <b>e2eqavn</b> build on Pytorch and provide command lined interface that allow user to train, test, and evaluate pipline.

# 2. Installation
```
pip install -U e2eqavn
```

# 3. Command line interface

Command | Function                                 |
--- |------------------------------------------|
e2eqavn train | - Training a model                       |
e2eqavn evaluate | - Evaluate the performance of a model    |
e2eqavn test | - Performs the inference of a text input |

## 3.1 Training model

```commandline
e2eqavn train --config [path_config_file]
```
This command starts a training pipeline that consists of retrieval or machine reading comprehension or both. If you want to train, you must provide path config yaml file. In this file config, you must setup the mode for training. 

<b> Example config file </b>:

```commandline
retrieval:
    is_train: true # turn on mode training retrieval
.....

reader:
    is_train: true # turn on mode training reader
```
You can see more detail parameter config in this [link]()
- Arguments:
```js
Required arguments:

  --config CONFIG_FILE, -c CONFIG_FILE        Path to the config file(architecture model, hyperparameter,..)

Optional arguments:
  
  --help, -h                                  Show this help message and exit.
```

- Example training CLI
```commandline
 e2eqavn train --config config.yaml
```

## 3.2 Evaluate pipeline

```commandline
e2eqavn evaluate [MODE] --config [path_config_file] 
                        --top_k_bm25 [INT_VALUE] # Optional, default value 10
                        --top_k_sbert [INT_VALUE] # Optional, default value 3
 
 
    MODE must in:
        - retrieval: Evaluate retrieval
        - reader: Evaluate reader
        - pipeline: Evalaute pipeline(retrieval + reader)           
```

This command enable user to evaluate model in 3 ways:
- Retrieval model: Calculate recall@k, precision@k, ndcg metric and log result to csv file
- Reader model: Calculate 2 metrics: Exact match, F1 for machine reading comprehension task 
- Pipeline model: Evaluate performance pipeline in Exact match and F1 score


<b>Arguments:</b>
```js
Required arguments:
   MODE                                               Selection mode for evalaute(retrieval, reader, pipeline)
  --config CONFIG_FILE                                Path to the config file.
  --top_k_bm25 TOP_K_BM25                             Top k document when retreival by BM25 algorithm
  --top_k_sbert TOP_K_SBERT                           Top k document when retrieval by SentenceTransformer Algorithm
  --logging_result_pipeline LOGGING_RESULT            Logging result predict to json file when mode equal pipeline

Optional arguments:
  
  --help, -h                                  Show this help message and exit.
```


## 3.2 Testing 
This command enable user for testing example with pipeline exist in local 

```commandline
e2eqavn evaluate [MODE] --config [path_config_file] 
                        --question [STRING_QUESTION]  
                        --top_k_bm25 [INT_VALUE] # Optional, default value 10
                        --top_k_sbert [INT_VALUE] # Optional, default value 3
                        --top_k_qa [INT_VALUE] # Optional, default value 1
 
 
    MODE must in:
        - retrieval: Evaluate retrieval
        - reader: Evaluate reader
        - pipeline: Evalaute pipeline(retrieval + reader)           
```

<b>Arguments:</b>
```js
Required arguments:
   MODE                                               Selection mode for evalaute(retrieval, reader, pipeline)
  --config CONFIG_FILE                                Path to the config file.
  --question QUESTION                                 Which question do you want to ask?
  --top_k_bm25 TOP_K_BM25                             Top k document when retreival by BM25 algorithm
  --top_k_sbert TOP_K_SBERT                           Top k document when retrieval by SentenceTransformer Algorithm
  --top_k_qa TOP_K_QA                                 Top k mrc result
    
Optional arguments:
  
  --help, -h                                  Show this help message and exit.
```

