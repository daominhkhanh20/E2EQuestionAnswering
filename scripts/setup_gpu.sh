apt update
apt upgrade -y
apt install vim -y
cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

git clone https://github.com/daominhkhanh20/E2EQuestionAnswering.git
git checkout -b develop
pip install -r requirements.txt

