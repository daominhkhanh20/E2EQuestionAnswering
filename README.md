# 1. Introduction

- <b>e2eqavn </b> is a 'simple' and 'easy to use' library that provides end to end pipeline for Question answering task. 

- <b>e2eqavn</b> build on Pytorch and provide command lined interface that allow user to train, test, and evaluate pipline.

# 2. Installation
```
pip install e2eqavn
```

# 3. Command line interface

Command | Function                                 |
--- |------------------------------------------|
e2eqavn train | - Training a model                       |
e2eqavn evaluate | - Evaluate the performance of a model    |
e2eqavn test | - Performs the inference of a text input |

## 3.1 Training model

```commandline
e2eqavn train --config [path_config]
```