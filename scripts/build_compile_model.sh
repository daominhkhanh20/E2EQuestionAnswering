mkdir -p model_compile/sbert
mkdir -p model_compile/qa
export PYTHONPATH=./
python3 example/triton/make_triton_complie_model_qa.py
python3 example/triton/make_triton_complie_model_sbert.py
