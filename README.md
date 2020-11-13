# 基于深度学习的中文语音识别系统

这里记录了我用[audier][1]的中文语音识别系统对thchs30进行全量训练的过程
生成的模型准确度为，可以直接拿来用。


## 1. Installation

sudo docker run -itd --gpus all --name tf1 -v /opt/Works.Linux:/Works -v /opt/download:/Downloads tensorflow/tensorflow:1.7.0-gpu-py3 bash

sudo exec -it tf1 bash

export LANG="C.UTF-8"

pip install scipy tqdm python_speech_features

pip install keras==2.1.6

ln -s /Downloads/data/THCHS-30/data_thchs30 data/data_thchs30

mkdir logs_lm/checkpoint


[1]:https://github.com/audier/DeepSpeechRecognition

