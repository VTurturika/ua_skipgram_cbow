FROM tensorflow/tensorflow:latest-py3
RUN pip install sklearn matplotlib scipy
WORKDIR /train
CMD tensorboard --logdir /train/models/