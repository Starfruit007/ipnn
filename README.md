# **IPNN - Indeterminate Probability Neural Network**  

不确定概率神经网络

Pytorch implementation of paper:  [Indeterminate Probability Theory](https://arxiv.org/abs/2303.11536)

## **Environment**

1. Our environment is: Python 3.9.12
    > pip install -r requirements.txt 

2. Our Hardware is: 1 TITAN RTX GPU with 24G.  

## **Quick Start**

1. Run IPNN on MNIST to check the unsupervised clustering results.
    - clustering results are stable：
        > python3 demo_mnist.py --data_path ./ --num_epoch 5 --split_shape 2 10 --train_epsilon 2 --learning_rate 1e-3
    - clustering results are unstable：
        > python3 demo_mnist.py --data_path ./ --num_epoch 5 --split_shape 2 10 --train_epsilon 1e-6 --learning_rate 1e-3

2. Run IPNN on more Datasets.  
    > python3 demo_mnist.py --num_epoch 10 --split_shape 2 2 5 --learning_rate 1e-3 --data_path ./ 
    > python3 demo_fashionmnist.py --num_epoch 10 --split_shape 2 2 5 --learning_rate 1e-3 --data_path ./  
    > python3 demo_cifar10.py --num_epoch  10- --split_shape 2 2 5 --learnng_rate 1e-4 --data_path ./cifar10/ 
    > python3 demo_stl10.py --num_epoch  10- --split_shape 2 2 5 --learnng_rate 1e-4 --data_path ./STL10/  

3. Run IPNN on `binary to decimal', incl. w/. and w/o. multi-degree classifcation task.
    > python3 demo_binary2decimal.py  --num_epoch=2 --split_shape 2 2 2 2 2 2 2 2 2 2 2 2 --num_classes 4096 --train_batch_size 4096  
    > python3 demo_binary2decimal.py  --num_epoch=2 --split_shape 2 2 2 2 2 2 2 2 2 2 --num_classes 1024 --train_batch_size 1024

## **Quick Results Check**

The logs of above commands are stored into log folder, you can easily access them if you do want.



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Starfruit007/ipnn&type=Date)](https://star-history.com/#Starfruit007/ipnn&Date)