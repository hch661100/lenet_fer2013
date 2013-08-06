lenet_fer2013
=============

lenet for face expression 2013

add the file create_face_batch.py, which is used to generate the data as lenet input.\n
Usage: python create_face_batch.py src_data dst_data\n
the src_data could be downloaded in the link https://www.kaggle.com/account/login?ReturnUrl=%2fc%2fchallenges-in-representation-learning-facial-expression-recognition-challenge%2fdownload%2ffer2013.tar.gz
After the dst_data is generated, please put it in the directory data.

modify the convolutional_mlp.py, which could be used to train the model for face expression 2013. you can run the gpurun.sh and cpurun.sh. 

The crossmapnorm is modified. After it is finished, it would add into the repository.

Update date: 2013/8/6

Chong Huang
