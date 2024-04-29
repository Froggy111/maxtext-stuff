# clone maxtext and JetStream
git clone https://github.com/google/JetStream
git clone https://github.com/google/maxtext

# set DEBIAN_FRONTEND to non-interactive
sudo ex +"%s@DPkg@//DPkg" -cwq /etc/apt/apt.conf.d/70debconf
sudo dpkg-reconfigure debconf -f noninteractive -p critical

sudo DEBIAN_FRONTEND=noninteractive apt update
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt upgrade
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade

# install libraries
cd /home/${user}/maxtext
DEBIAN_FRONTEND=noninteractive bash setup.sh
cd /home/${user}
cd JetStream
pip install .
cd /home/${user}

pip install -U torch
pip install -U transformers
pip install -U huggingface_hub[hf_transfer]
export HF_HUB_ENABLE_HF_TRANSFER=1