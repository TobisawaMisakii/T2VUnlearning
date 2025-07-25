#!/bin/bash

# update apt source and install downloader aria2, git and git-lfs
apt update && apt install -y aria2 git git-lfs

# install ifs for git
git lfs install

# download hfd.sh (huggingface downloader)
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

# move hfd.sh to user local bin
mv hfd.sh /usr/local/bin/hfd

# setup environment variable
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

