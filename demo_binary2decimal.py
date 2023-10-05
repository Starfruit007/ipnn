import torch
import numpy as np
import argparse
from src.ipnn import IPNN
from src.trainer import Exp_Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(1)

def data_generator(num_classes,split_shape,in_shape = 13, multi_degree = False):
    ''' genenrate binary vectors and its corresponding decimal labels.'''
    numbers = 500
    y = np.linspace(0, num_classes-1,num_classes, dtype=int)
    vfunc = np.vectorize(lambda i: '{0:08b}'.format(i))
    x_bin = vfunc(y)

    for _ in x_bin:
        tmp = np.array(list('00000000'+_)[-in_shape:],dtype=float) # make binary string length to 13
        if _ == '00000000': x = tmp
        else:
            x = np.vstack((x,tmp))

    x_train_arr = np.tile(x,(numbers,1)) # repeat the input samples
    y_train_arr = np.tile(y,numbers) # repeat the output labels

    x_train, y_train = torch.tensor(x_train_arr,dtype=torch.float), torch.tensor(y_train_arr,dtype=torch.int64) # data type format
    x_eval, y_eval = torch.tensor(np.tile(x,(10,1)),dtype=torch.float), torch.tensor(np.tile(y,10),dtype=torch.int64)

    if multi_degree: # this is for multi-dgree classification task.
        if all(np.array(split_shape) == 2):
            # Input samples are additionally labeled with $Y^{i} \in \{0,1\}$ for $i^{th}$ bit is 0 or 1, respectively. 
            # $Y^{i}$ corresponds to sub joint sample space $\Lambda^{i}$ with split shape $\{M_{i}=2\}, i = 1,2,\dots12$.
            y_train = np.hstack((np.expand_dims(y_train,axis = -1),x_train[:,-len(split_shape):].long()))
        else:
            print('multi-dgree classification task is not enabled, currently, the code only support split shape is [2,2,...,2].')

    print('classification classes is: {}'.format(y_train.max()+1))
    
    return x_train, y_train, x_eval, y_eval


class MODEL_IPNN(torch.nn.Module):
    def __init__(self,in_shape,args):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_shape, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, sum(args.split_shape)),
        )
        self.classifier.apply(self.init_weights) # in order to avoid local minimum at training begining.
        self.ipnn = IPNN(forget_num = {'training':args.train_forget_num,'prediction':args.prediction_forget_num,'mutual independent': args.mutual_independence_forget_num}, 
                        stable_num =  {'training':args.train_epsilon,'prediction':args.prediction_epsilon,'mutual independent':args.mutual_independence_epsilon})
        self.split_shape = args.split_shape
        self.num_classes = args.num_classes

        
    def forward(self, x, labels):
        x = x.to(device)
        if labels is not None: labels = labels.to(device)
        logits = self.classifier(x)
        if labels is not None and len(labels.shape) > 1:
            tmp = torch.nn.functional.one_hot(labels[:,0], self.num_classes).float().to(device)
            y_tures = [tmp]
            
            NNN  = min(labels.shape[-1], len(self.split_shape)+1)
            select_variables = [list(range(NNN-1))]

            for i in range(1,NNN):
                tmp = torch.nn.functional.one_hot(labels[:,i],2).detach().cpu().numpy()
                tmp = torch.as_tensor(torch.from_numpy(tmp), dtype=torch.float32).to(device)
                y_tures.append(tmp)
                select_variables.append([i-1])

        elif labels is not None and len(labels.shape) == 1:
            tmp = torch.nn.functional.one_hot(labels,self.num_classes).detach().cpu().numpy()
            tmp = torch.as_tensor(torch.from_numpy(tmp), dtype=torch.float32).to(device)
            y_tures = [tmp]
            select_variables = None
        else:
            y_tures, select_variables = None, None

        outputs = self.ipnn(logits,y_tures,select_variables = select_variables,split_shape = self.split_shape)
        loss = sum(outputs['losses'])
        return loss, outputs

    def init_weights(self,m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.uniform_(m.weight,a=-0.3, b=0.3)
            torch.nn.init.uniform_(m.bias,a=-0.3, b=0.3)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x, self.y = x, y
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)


def parse_args():
    parser = argparse.ArgumentParser(description="Set Parameters for IPNN - Indeterminate Probability Neural Network.")
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=5,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3, 
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
        default=[2,2,2,2,2,2,2,2,2,2, 2,2], 
        help="split the output neurons into defined shape.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4096, 
        help="number of to be claissified binaries",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4096, 
        help="train batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256, 
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


def main(multi_degree = True):

    args = parse_args()

    in_shape = 13

    x_train, y_train, x_eval, y_eval  = data_generator(args.num_classes,args.split_shape,in_shape,multi_degree)

    train_dataset = MyDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.train_batch_size,num_workers = 0, shuffle = True)
    eval_dataset = MyDataset(x_eval, y_eval)
    eval_loader = torch.utils.data.DataLoader(dataset = eval_dataset, batch_size = args.eval_batch_size,num_workers = 0, shuffle = False)

    # Model Initialization
    model = MODEL_IPNN(in_shape,args)
    model.to(device)
    exp = Exp_Trainer(model,train_loader,eval_loader,args.num_epoch,args.learning_rate, args.weight_decay)
    recorder_dict = exp.exp_start()

    return recorder_dict



if __name__ == "__main__":
    round = 10
    accs1,accs2 = [],[]
    for i in range(round):
        print('Round {}/{} modelling:'.format(i+1,round))

        print('with multi-degree classification task training: ')
        recorder_dict = main(multi_degree = True)
        accs1.append(float(recorder_dict['results']['accuracy'][0]))
        print('w/. multi... accs: {}, mean: {}, std: {}.'.format(accs1,np.mean(accs1),np.std(accs1)))

        print('\n without multi-degree classification task training: ')
        recorder_dict = main(multi_degree = False)
        accs2.append(float(recorder_dict['results']['accuracy'][0]))
        print('w/o. multi... accs: {}, mean: {}, std: {}.'.format(accs2,np.mean(accs2),np.std(accs2)))

    print("hello world~")