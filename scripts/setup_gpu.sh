cd /workspace
git clone https://github.com/daominhkhanh20/E2EQuestionAnswering.git

apt update
apt upgrade -y
apt install vim -y
apt install tmux -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

export PATH=/root/miniconda3/bin:$PATH

export PATH=/workspace/miniconda3/bin:$PATH
export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"
export PATH=/root/miniconda3/bin:$PATH

cd /workspace/E2EQuestionAnswering
git checkout develop
git checkout develop_pipeline
pip install python-dotenv
pip install .
python3 -c "import nltk;nltk.download('punkt')"

tmux new -s train_rcm
export PYTHONPATH=./
python3 test/mrc/training_mrc.py

kaggle datasets init -p model
kaggle datasets create -p model -r zip 

e2eqavn train --config config/train_qa1.yaml
e2eqavn train --config config/train_qa2.yaml

rm model/qa/*/*/sche*
rm model/qa/*/*/optim*
rm model/qa/*/*/train*


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
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
git config --global user.email "khanhc1k36@gmail.com"
git config --global user.name "daominhkhanh20"

daominhkhanhbk20/triton_backend
bash -c "apt update;apt install -y wget;DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;mkdir -p ~/.ssh;cd $_;chmod 700 ~/.ssh;echo ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKvhxW8+MW8HydnawgFvHCMJ575EyQBdMlIQ1IrmBZaZ khanhc1k36@gmail.com > authorized_keys;chmod 700 authorized_keys;service ssh start;sleep infinity"


