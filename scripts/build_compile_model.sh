#mkdir -p model_compile/sbert
#mkdir -p model_compile/qa
#export PYTHONPATH=./
#python3 example/triton/make_triton_complie_model_qa.py
python3 example/triton/make_triton_complie_model_sbert.py

export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
kaggle datasets init -p model_compile
MODEL_VERSION='model-compile-v4'
sed -i "s/INSERT_TITLE_HERE/$MODEL_VERSION/g" model_compile/dataset-metadata.json
sed -i "s/INSERT_SLUG_HERE/$MODEL_VERSION/g" model_compile/dataset-metadata.json
kaggle datasets create -p model_compile -r zip
