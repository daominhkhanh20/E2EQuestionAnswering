mkdir -p model_compile/sbert
mkdir -p model_compile/qa
export PYTHONPATH=./
python3 example/triton/make_triton_complie_model_qa.py
python3 example/triton/make_triton_complie_model_sbert.py --from_mongo $1
export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
kaggle datasets init -p model_compile
sed -i 's/INSERT_TITLE_HERE/model_compile_v3/g' model_compile/dataset-metadata.json
sed -i 's/INSERT_SLUG_HERE/model_compile_v3/g' model_compile/dataset-metadata.json
kaggle datasets create -p model_compile -r zip
