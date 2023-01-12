

# **IPNN - Indeterminate Probability Neural Network**

Pytorch implementation of ICML2023 paper:  **Indeterminate Probability Neural Network**,  
By Yang Tao

## **Abstract**

    We propose a new general model called IPNN - Indeterminate Probability Neural Network, 
    which combines neural network and classical probability theory together. In the classical probability theory, 
    the calculation of probability is based on the occurrence of events, which is hardly used in current neural networks. 
    In this paper, the output of neural network is defined as probability events, and based on the statistical analysis 
    of these events, the inference model for classification task is deduced. IPNN shows new property: It can perform unsupervised clustering 
    while doing classification. Besides, it is able of making very large classification with very small neural network, e.g. model with 100 output nodes can classify 10 billion categories. 
    Theoretical advantages are reflected in experimental results.  

## **Environment**

1. Our environment is: Python 3.9.12
    > pip install -r requirements.txt 

## **Quick Start**

1. Run IPNN on MNIST to check the unsupervised clustering results.
    - clustering results are stable：
        > python3 demo_mnist.py --mnist_data_path ./ --num_epoch 5 --split_shape 2 10 --train_epsilon 2
    - clustering results are unstable：
        > python3 demo_mnist.py --mnist_data_path ./ --num_epoch 5 --split_shape 2 10 --train_epsilon 1e-6

3. Run IPNN on `binary to decimal', incl. w/. and w/o. multi-degree classifcation task.
    > python3 demo_binary2decimal.py  --num_epoch=2 --split_shape 2 2 2 2 2 2 2 2 2 2 2 2 --num_classes 4096 --train_batch_size 4096  
    > python3 demo_binary2decimal.py  --num_epoch=2 --split_shape 2 2 2 2 2 2 2 2 2 2 --num_classes 1024 --train_batch_size 1024

## **Quick Results Check**

The logs of above commands are stored into log folder, you can easily access them if you do want.
