cd /workspace
git clone https://github.com/daominhkhanh20/E2EQuestionAnswering.git

apt update
apt upgrade -y
apt install vim -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

export PATH=/workspace/miniconda3/bin:$PATH

cd E2EQuestionAnswering
git checkout -b develop3
pip install -r requirements.txt
