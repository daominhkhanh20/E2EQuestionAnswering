export PYTHONPATH=./

export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
rm model/qa/*/optimizer.pt
kaggle datasets init -p model/qa
sed -i 's/INSERT_TITLE_HERE/model-compile-qa/g' model/qa/dataset-metadata.json
sed -i 's/INSERT_SLUG_HERE/model-compile-qa/g' model/qa/dataset-metadata.json
kaggle datasets create -p model/qa -r zip