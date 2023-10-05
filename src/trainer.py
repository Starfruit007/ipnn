import torch
import numpy as np
from src.utils import IPNNEvaluation   


class Exp_Trainer():

    def __init__(self,model,train_loader,eval_loader,num_epoch,learning_rate, weight_decay) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.num_epoch = num_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.ipe = IPNNEvaluation()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.learning_rate, weight_decay = self.weight_decay)


    def train_one_epoch(self,epoch):

        for step, (x, labels) in enumerate(self.train_loader):

            loss, outputs = self.model(x,labels)
            results = self.ipe.main(outputs, labels, main_label_index = 0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            act_acc = results['actual_accuracy'][0]
            acc =  results['accuracy'][0]
            average_loss = results['average_loss']
            print(f'\r epoch {epoch+1}/{self.num_epoch} - curr/avg acc: {act_acc}/{acc}\
                - curr/avg loss: {loss.item():>7f}/{average_loss}, [{step+1:>5d}/{len(self.train_loader):>5d}]', end='')
        print('\n')


    def test_one_epoch(self):

        for step, (x, labels) in enumerate(self.eval_loader):
            with torch.no_grad():
                _, outputs = self.model(x,labels = None)
                results = self.ipe.main(outputs, labels, main_label_index = 0)
                
                act_acc = results['actual_accuracy'][0]
                acc = results['accuracy'][0]
                act_unique_acc, unique_acc = results['actual_unique_accuracy'], results['unique_accuracy']
                print(f'\r prediction - curr/avg acc: {act_acc}/{acc}\
                    - curr/avg unique acc: {act_unique_acc}/{unique_acc}, [{step+1:>5d}/{len(self.eval_loader):>5d}]', end='')
        print('\n')

        return results

    def exp_start(self):
        total_losses = []
        total_accuracy = []
        for epoch in range(self.num_epoch):
            self.ipe.reset_recorder()
            self.train_one_epoch(epoch)

            total_losses.append(self.ipe.recorder_dict['total_losses'])
            total_accuracy.append(self.ipe.recorder_dict['results_all']['actual_accuracy'])

            self.ipe.reset_recorder()
            results = self.test_one_epoch()

        recorder_dict = self.ipe.recorder_dict
        recorder_dict['results'] = results
        recorder_dict['total_losses'] = total_losses
        recorder_dict['total_accuracy'] = total_accuracy


        return recorder_dict
