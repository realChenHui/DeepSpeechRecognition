
sudo docker run -itd --gpus all --name tf1 -v /opt/Works.Linux:/Works -v /opt/download:/Downloads tensorflow/tensorflow:1.7.0-gpu-py3 bash

export LANG="C.UTF-8"
pip install scipy tqdm python_speech_features
pip install keras==2.1.0

ln -s /Downloads/data/THCHS-30/data_thchs30 data/data_thchs30


