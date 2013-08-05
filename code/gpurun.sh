#!/bin/sh

export PATH=/home/zhaonan/environment/python-2.7/bin:/home/software/cuda-5.0/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=/home/software/cuda-5.0/lib:/home/software/cuda-5.0/lib64:/home/zhaonan/environment/python-2.7/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/zhaonan/environment/lib/python2.7/site-packages:$PYTHONPATH

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python convolutional_mlp.py
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test_convnet.py
