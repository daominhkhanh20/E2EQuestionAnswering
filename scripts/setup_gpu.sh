cd /workspace
git clone https://github.com/daominhkhanh20/E2EQuestionAnswering.git

apt update
apt upgrade -y
apt install vim -y
apt install tmux -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

export PATH=/workspace/miniconda3/bin:$PATH

export KAGGLE_USERNAME='daominhkhanh'
export KAGGLE_KEY="53d2021e6812290870fc3520cbeee5ea"

cd E2EQuestionAnswering
git checkout -b develop3
pip install -r requirements.txt

bash -c "apt update;apt install -y wget;DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;mkdir -p ~/.ssh;cd $_;chmod 700 ~/.ssh;echo ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKvhxW8+MW8HydnawgFvHCMJ575EyQBdMlIQ1IrmBZaZ khanhc1k36@gmail.com > authorized_keys;chmod 700 authorized_keys;service ssh start;sleep infinity"