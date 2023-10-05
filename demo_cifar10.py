import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pandas as pd
import argparse

from src.ipnn import IPNN
from src.trainer import Exp_Trainer
from transformers import ResNetForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(1)


class ResetNet50(torch.nn.Module):
    def __init__(self,split_shape):
        super().__init__()
        self.resnet = ResNetForImageClassification.from_pretrained("resnet-50").resnet
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, sum(split_shape)),
        )



    def forward(self, image):
        x = self.resnet(image).pooler_output
        outputs = self.classifier(x)

        return outputs
    

    
class MODEL_IPNN(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.classifier = self.classifier = ResetNet50(args.split_shape)
        self.ipnn = IPNN(forget_num = {'training':args.train_forget_num,'prediction':args.prediction_forget_num,'mutual independent': args.mutual_independence_forget_num}, 
                        stable_num =  {'training':args.train_epsilon,'prediction':args.prediction_epsilon,'mutual independent':args.mutual_independence_epsilon})
        self.split_shape = args.split_shape


    def labeling(self,labels):
        if labels is None: return None,None
        y_true = torch.nn.functional.one_hot(labels,10).float().to(device)
        y_trues = [y_true]
        select_variables = None # used for multi-degree classification task
        return y_trues, select_variables

    def forward(self, images, labels = None):
        images = images.to(device)
        if labels is not None: labels = labels.to(device)   
        logits = self.classifier(images)
        y_trues,select_variables = self.labeling(labels)
        
        outputs = self.ipnn(logits,y_trues,select_variables,split_shape = self.split_shape)
        losses = outputs['losses']
        loss = sum(losses)

        return loss, outputs

def print_unsupervised_cluster_results(recorder_dict, random_variable_index = 0):

    num_classes = 10

    total_labels = recorder_dict['total_labels'][:,0]
    total_vars_arg = recorder_dict['total_vars_arg']

    rs = {}
    for i in range(len(total_labels)):
        outs, lb = total_vars_arg[i], total_labels[i]
        ky = outs[random_variable_index]
        if ky not in rs: rs[ky] = []
        rs[ky].append(lb)



    cluster_results = np.zeros((len(rs),num_classes))
    for ky in rs:
        tmp = np.unique(rs[ky],return_counts=True)
        cluster_results[ky,tmp[0]] = tmp[1]

    print('unsupervised cluster results of random variable {} is:'.format(random_variable_index))
    x = np.linspace(0,num_classes-1,num_classes,dtype = int)
    table = pd.DataFrame(cluster_results, columns = x,dtype=int)
    print(table)
    


def parse_args():
    parser = argparse.ArgumentParser(description="Set Parameters for IPNN - Indeterminate Probability Neural Network.")
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=10,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4, 
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0, 
        help="weight decay",
    )
    parser.add_argument(
        "--split_shape",
        nargs='+', 
        type=int,
        default=[2,2,5], 
        help="split the output neurons into defined shape.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='./cifar10/', 
        help="dataset path.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64, 
        help="train batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128, 
        help="eval batch size",
    )
    parser.add_argument(
        "--train_forget_num",
        type=int,
        default=5, 
        help="forget number T for training",
    )
    parser.add_argument(
        "--prediction_forget_num",
        type=int,
        default=5, 
        help="forget number T for prediction",
    )
    parser.add_argument(
        "--mutual_independence_forget_num",
        type=int,
        default=5, 
        help="forget number T for mutual independence loss",
    )
    parser.add_argument(
        "--train_epsilon",
        type=float,
        default=1e-6, 
        help="epsilon (or stable number) for training",
    )
    parser.add_argument(
        "--prediction_epsilon",
        type=int,
        default=1e-6, 
        help="epsilon (or stable number) for prediction",
    )
    parser.add_argument(
        "--mutual_independence_epsilon",
        type=int,
        default=1e-6, 
        help="epsilon (or stable number) for mutual independence loss",
    )

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    # Download the MNIST Dataset
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    # define transforms
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    train_dataset = datasets.cifar.CIFAR10(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.cifar.CIFAR10(root=args.data_path, train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.train_batch_size,num_workers = 0, shuffle = True)
    eval_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.eval_batch_size,num_workers = 0, shuffle = False)

    # Model Initialization
    model = MODEL_IPNN(args)
    model.to(device)

    exp = Exp_Trainer(model,train_loader,eval_loader,args.num_epoch,args.learning_rate, args.weight_decay)
    recorder_dict = exp.exp_start()
    print_unsupervised_cluster_results(recorder_dict, random_variable_index = 0)
    # print_unsupervised_cluster_results(recorder_dict, random_variable_index = 1)
    return recorder_dict

if __name__ == "__main__":
    round = 10
    accs = []
    for i in range(round):
        print('Round {}/{} modelling:'.format(i+1,round))
        recorder_dict = main()
        accs.append(float(recorder_dict['results']['accuracy'][0]))
        print('accs: {}, mean: {}, std: {}.'.format(accs,np.mean(accs),np.std(accs)))

    print("hello world~")