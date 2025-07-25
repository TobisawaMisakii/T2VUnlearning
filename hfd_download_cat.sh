#!/bin/bash

# source ~/.bashrc

# 在AutoDL中，/root/autodl-tmp存放在数据盘，方便存放模型参数
# 此处以 google/ddpm-cat-256 为例，可以将两处 google/ddpm-cat-256 和 ddpm-cat-256 修改为对应模型文件
cd /root/autodl-tmp && hfd THUDM/CogVideoX-5b --local-dir /root/autodl-tmp/models/cogvideox-5b --tool aria2c -x 8 -j 8
