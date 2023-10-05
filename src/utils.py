import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from copy import deepcopy

class IPNNEvaluation:
    def __init__(self) -> None:
        self.recorder_dict = {}

    def reset_recorder(self):
        self.recorder_dict = {}
        
    def torch_to_numpy(self, outputs):
        ''' convert type of model outputs from tensor to numpy.'''
        outputs_numpy = {}
        for ky in outputs:
            if isinstance(outputs[ky],list):
                outputs_numpy[ky] = [_.detach().cpu().numpy() for _ in outputs[ky]]
            elif isinstance(outputs[ky],dict):
                outputs_numpy[ky] = {}
                for subky in outputs[ky]:
                    outputs_numpy[ky][subky] = outputs[ky][subky].detach().cpu().numpy() 
        return outputs_numpy

    def compare(self, preds,label):
        ''' comparison between predictions and labels. '''
        preds_label = np.argmax(preds,axis = -1)
        true_nums = np.sum(preds_label == label)
        return preds_label, true_nums

    def accuracy_calc(self, outputs_numpy, labels_numpy):
        '''accuracy calcation: prediction is the indexes of maximum posterior.
        '''
        true_nums_dict = {}
        preds_label_dict = {}
        for i in outputs_numpy['probs_pred']: # for multi-degree classification task.
            preds = outputs_numpy['probs_pred'][i]
            if i==0 or labels_numpy.shape[-1] == len(list(outputs_numpy['probs_pred'].keys())):
                label = labels_numpy[:,i]
                preds_label, true_nums = self.compare(preds,label)
            elif i > 0: 
                label = labels_numpy[:,0]
                preds_label, true_nums = self.compare(preds,label)
                true_nums = 0
            true_nums_dict[i] = true_nums
            preds_label_dict[i] = preds_label

        return true_nums_dict, preds_label_dict

    def unique_label(self, outputs_numpy):
        ''' This is used for unique accuracy calculation, which aims to find out how many labels have an unique joint sample point.
            HOW TO:
                Findout the joint sample point with maximum porobability,
                and assign joint sample point indexes to this input sample.
                This is needed for evaluation function of adjusted_rand_score.
        '''
        vars_numpy = outputs_numpy['variables_all']
        vars_arg = np.array([np.argmax(_,axis = -1) for _ in vars_numpy]).T

        # unique variable combination id
        for i in range(vars_arg.shape[1]):
            if i == 0:  
                vars_id = deepcopy(vars_arg[:,i])
                continue
            vars_id += vars_arg[:,i]*10**i

        return vars_id, vars_arg
    

    def main(self,outputs, labels, main_label_index = 0):
        ''' evaluation results will be returned.
            Besides, some varialbes are recorded in self.recorder_dict for later analysis use.'''
        outputs_numpy = self.torch_to_numpy(outputs)
        labels_numpy = labels.detach().cpu().numpy()

        batch_size = labels_numpy.shape[0]
        if len(labels_numpy.shape)==1: # shape higher than 1 is the case of multi-degree classification task.
            labels_numpy = np.expand_dims(labels_numpy,axis = -1)
            
        label = labels_numpy[:,main_label_index] # main results of multi-degree classification task.


        true_nums_dict, preds_label_dict = self.accuracy_calc(outputs_numpy, labels_numpy) # actual accuracy calculation of the input batch.
        act_acc = {ky:true_nums_dict[ky]/batch_size for ky in true_nums_dict}

        vars_id, vars_arg = self.unique_label(outputs_numpy) # assign joint sample point with maximum probability to sample
        act_unique_acc = adjusted_rand_score(vars_id,label)  # actual accuracy calculation of the input batch from the perspective of joint sample points.

        # record the evaluation results into self.recorder_dict
        losses_numpy = np.array(outputs_numpy['losses'])
        self.recorder(labels_numpy,vars_id,vars_arg,losses_numpy,preds_label_dict,true_nums_dict)



        total_number = self.recorder_dict['total_labels'].shape[0]
        # unique accuray: find out how many labels have an unique joint sample point.
        unique_acc = adjusted_rand_score(self.recorder_dict['total_vars_id'],self.recorder_dict['total_labels'][:,main_label_index])  # overall accuracy calculation.
        acc = {ky:self.recorder_dict['total_true_nums_dict'][ky]/total_number for ky in self.recorder_dict['total_true_nums_dict']} # overall accuracy calculation.
        
        if list(losses_numpy) != []:
            average_loss = np.sum(self.recorder_dict['total_losses'])/self.recorder_dict['total_losses'].shape[0]
        else: average_loss = 0
        
        results = dict(
            accuracy = acc,
            unique_accuracy = unique_acc,
            actual_accuracy = act_acc,
            actual_unique_accuracy = act_unique_acc,
            average_loss = average_loss
        )
        self.record_results(results) # record evaluation results into self.recorder_dict 
        results = self.format(results) # evaluation results rounding
        return results

    def record_results(self,results):
        ''' record evaluation results. '''
        if 'results_all' not in self.recorder_dict:
            self.recorder_dict['results_all'] = {}
        for ky in results:
            if ky not in self.recorder_dict['results_all']:      self.recorder_dict['results_all'][ky] = []
            if isinstance(results[ky],dict):
                tmp = list(results[ky].values())
            else:
                tmp = results[ky]
            self.recorder_dict['results_all'][ky].append(tmp)

   

    def recorder(self,labels_numpy,vars_id,vars_arg,losses_numpy,preds_label_dict,true_nums_dict):
        ''' record some variables for later anlaysis use.'''
        if 'total_labels' not in self.recorder_dict:
            self.recorder_dict['total_labels'] = labels_numpy
        else:
            self.recorder_dict['total_labels'] = np.vstack((self.recorder_dict['total_labels'],labels_numpy))

        if 'total_vars_id' not in self.recorder_dict:
            self.recorder_dict['total_vars_id'] = vars_id
        else:
            self.recorder_dict['total_vars_id'] = np.hstack((self.recorder_dict['total_vars_id'],vars_id))

        if 'total_vars_arg' not in self.recorder_dict:
            self.recorder_dict['total_vars_arg'] = vars_arg
        else:
            self.recorder_dict['total_vars_arg'] = np.vstack((self.recorder_dict['total_vars_arg'],vars_arg))

        if 'total_losses' not in self.recorder_dict:
            self.recorder_dict['total_losses'] = losses_numpy
        else:
            self.recorder_dict['total_losses'] = np.vstack((self.recorder_dict['total_losses'],losses_numpy))

        if 'total_preds_label_dict' not in self.recorder_dict:
            self.recorder_dict['total_preds_label_dict'] = {ky:preds_label_dict[ky] for ky in preds_label_dict} # avoid share same memory address
        else:
            for ky in preds_label_dict:
                self.recorder_dict['total_preds_label_dict'][ky] = np.hstack((self.recorder_dict['total_preds_label_dict'][ky],preds_label_dict[ky]))
        
        if 'total_true_nums_dict' not in self.recorder_dict:
            self.recorder_dict['total_true_nums_dict'] = {ky:true_nums_dict[ky] for ky in true_nums_dict} # avoid share same memory address
        else:
            for ky in preds_label_dict:
                self.recorder_dict['total_true_nums_dict'][ky] = self.recorder_dict['total_true_nums_dict'][ky] + true_nums_dict[ky]

    def format(self, results):
        ''' round the evaluation results '''
        results['actual_accuracy'] = [f'{_:>5f}' for _ in results['actual_accuracy'].values()]
        results['accuracy'] = [f'{_:>5f}' for _ in results['accuracy'].values()]
        
        actual_unique_accuracy = results['actual_unique_accuracy']
        results['actual_unique_accuracy'] = f'{actual_unique_accuracy:>5f}'

        unique_accuracy = results['unique_accuracy']
        results['unique_accuracy'] = f'{unique_accuracy:>5f}'

        average_loss = results['average_loss']
        results['average_loss'] = f'{average_loss:>5f}'
        return results
