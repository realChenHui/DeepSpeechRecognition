# 基于深度学习的中文语音识别系统

这里是我对[audier][1]的中文语音识别系统进行改造的记录，主要改动如下：
- 添加CBHG声学模型
- 添加n-gram语言模型

详细内容可看我的blog

## 1. Installation

sudo docker run -itd --gpus all --name tf1 -v /opt/Works.Linux:/Works -v /opt/download:/Downloads tensorflow/tensorflow:1.7.0-gpu-py3 bash

sudo exec -it tf1 bash

export LANG="C.UTF-8"

pip install scipy tqdm python_speech_features

pip install keras==2.1.6

ln -s /Downloads/data/THCHS-30/data_thchs30 data/data_thchs30

mkdir logs_lm/checkpoint


[1]:https://github.com/audier/DeepSpeechRecognition

