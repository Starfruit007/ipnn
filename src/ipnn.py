import torch

EINSUM_CHAR = 'ijklmnopqrstuvwIJKLMNOPQRSTUVW' # a,b,y is forbidden to be defined within this char.

class IPNN(torch.nn.Module):
    ''' 
        Random Variable Explanation
            Input sample: Xi
            Model outputed random variables: A1, A2, ..., AN
            Label of input sample: Yi
    '''
    def __init__(self,forget_num = {'training':5,'prediction':50,'mutual independent':5}, 
                stable_num =  {'training':1e-6,'prediction':1e-6,'mutual independent':1e-6}):
        super().__init__()

        self.forget_num = forget_num
        self.stable_num = stable_num

        self.recorder = {'training':{'num_y_joint':{},'num_joint':{},'num_y_joint_memory':{},'num_joint_memory':{},'prob_y_joint':{},'probability':{}},
                        'prediction':{'num_y_joint':{},'num_joint':{},'num_y_joint_memory':{},'num_joint_memory':{},'prob_y_joint':{},'probability':{}},
                        'mutual independent':{'num_variables':None,'num_joint':None,'num_variables_memory':[],'num_joint_memory':[]}}


    def forward(self, logits, y_trues = None, select_variables = None, split_shape = None, independence_variables = None):
        '''
            calcalation of the loss and the posterior P^{A}(Yi|Xi).
            Args:
                logits: output nodes of neural network
                y_trues: list - multi-degree classification
                    label of input samples in a mutli-degree level.
                select_variables: sub joint space selection indexes, which corresponds to y_trues. 
                    If it is set to None, the multi-degree classification will not perform.
                split_shape: shape of join sample space
                independence_variables: indexes of random variables by adding additional mutual independence loss.
        '''
        if select_variables is None:
            select_variables = [list(range(len(split_shape)))]     
        batch_size = logits.shape[0]
        assert len(split_shape) < len(EINSUM_CHAR), \
                'Currently torch.einsum used in IPNN does not support len(split_shape):{} higher than len(EINSUM_CHAR): {}.'.format(len(split_shape), len(EINSUM_CHAR))   
        if split_shape is not None and len(split_shape)>1:
            # split outputs of neural network into N parts.
            logits = torch.split(logits, split_shape, dim = -1)
            
        else:
            # it means only one random variable available
            logits = [logits]
        
        # The splited nodes via softmax becomes random variables: A1, A2, ..., AN.
        # the nodes of each random variable are regarded as collectively exhuastive and exclusive events.
        # shape of each random variables are: [[batch_size, M1], [batch_size, M2], ..., [batch_size, MN]]
        variables_all = [torch.softmax(_,dim = -1) for _ in logits]

        losses = []
        # for loop is used for multi-degree classification task
        for i in range(len(select_variables)):
            # for multi degree classificaiton (or clustering) task, use predefined random variables.
            variables = [variables_all[_] for _ in select_variables[i]]


            # joint event probability for each joint sample point: (A1, A2, ..., AN).
            # shape of joint_variables is: [batch_size, M1, M2, ..., MN]
            joint_variables = self.joint_sample_space(variables)
            
            # observation of relationship between joint events and labels
            # that is calculation of contional probability P(Yi|A1,A2,...,AN)
            # shape of prob_y_joint P(Yi|A1,A2, ..., AN) is: [number of labels, M1, M2, ..., MN]
            if y_trues is not None:
                self.recorder['training']['prob_y_joint'][i] = self.observation('training', i, joint_variables,y_trues[i])
                self.recorder['prediction']['prob_y_joint'][i] = self.observation('prediction', i, joint_variables,y_trues[i]) # forget number can be different from training process.
            
            # calculation of the posterior P^{A}(Yi|Xi): infer the label Yi of input sample Xi
            # shape of the posterior P^{A}(Yi|Xi): [batch_size, number of labels]
            self.recorder['training']['probability'][i] = self.inference(self.recorder['training']['prob_y_joint'][i], joint_variables)
            self.recorder['prediction']['probability'][i] = self.inference(self.recorder['prediction']['prob_y_joint'][i], joint_variables)

            # cross entropy loss
            if y_trues is not None:
                probs_sum = torch.sum(torch.mul(self.recorder['training']['probability'][i],y_trues[i]),axis = 1)
                loss = torch.sum(torch.sum(-torch.log(probs_sum),dim = -1)) / batch_size
                losses.append(loss)


        if independence_variables is not None:
            variables = [variables_all[_] for _ in independence_variables]
            loss = self.mutual_independence_loss(variables)
            losses.append(loss)

        self.detach()
       
        return dict(variables_all = variables_all,
                    losses = losses,
                    probs_train = self.recorder['training']['probability'],
                    probs_pred = self.recorder['prediction']['probability'],)


    def joint_sample_space(self,variables, batched = True):
        ''' caclulation probability of joint sample spaces of all random variables (A1,A2, ..., AN) given input sample Xi.
            Assumption: Given one input sample Xi from mini batch, 
                        all random variables A1,A2, ..., AN are conditionally mutually independent.'''
        for i in range(len(variables)):
            if i == 0 : 
                joint_variables = variables[i]
            else:
                if batched:
                    # shape of joint_variables is: [batch_size, M1, M2, ..., MN]
                    r_ = EINSUM_CHAR[:joint_variables.dim()-1]
                    joint_variables = torch.einsum('b{},ba->b{}a'.format(r_,r_),joint_variables,variables[i])
                else:
                    # shape of joint_variables is: [M1, M2, ..., MN]
                    r_ = EINSUM_CHAR[:joint_variables.dim()]
                    joint_variables = torch.einsum('{},a->{}a'.format(r_,r_),joint_variables,variables[i])
        
        return joint_variables


    def observation(self,ky, i,joint_variables,y_true):
        ''' calculation of contional probability P(Yi|A1,A2,...,AN) '''
        r_ = EINSUM_CHAR[:joint_variables.dim()-1]
        num_y_joint_current = torch.einsum('b{},by->y{}'.format(r_,r_),joint_variables,y_true) 
        num_joint_current = torch.sum(joint_variables,dim = 0)

        # update self.num_y_joint[i] and self.num_joint[i]
        prob_y_joint = self.num_prob_recorder(ky, i = i, num_y_joint=num_y_joint_current,num_joint=num_joint_current)
        return prob_y_joint
         

    def inference(self,prob_y_joint, joint_variables):
        ''' calculation of the posterior P^{A}(Yi|Xi) '''
        # probs = torch.mul(torch.unsqueeze(prob_y_joint,0),torch.unsqueeze(joint_variables,1)) # high memory consumption
        # tmp_shape = list(range(2,probs.dim()))
        # probs_sum = torch.sum(probs,tmp_shape)

        r_ = EINSUM_CHAR[:joint_variables.dim()-1]
        probs_sum = torch.einsum('y{},b{}->by'.format(r_,r_),prob_y_joint,joint_variables)
        return probs_sum

    def num_prob_recorder(self, ky = 'training', i=None, num_y_joint=None,num_joint=None):
        ''' record the denominator and numerator of conditional probability P(Yi|A1,A2,...,AN), respectively.
            And forget the previous results according to forget number: self.forget_num[ky].
            
            Args:
                ky: default is 'training', it support to set different record parameters between training and prediction.
                i: indicator of multi-degree classification task.
                num_y_joint: numerator of conditional probability P(Yi|A1,A2,...,AN)
                num_joint: denominator of conditional probability P(Yi|A1,A2,...,AN)
        '''

        # forgetting process
        if not i in self.recorder[ky]['num_y_joint_memory']:
            self.recorder[ky]['num_y_joint_memory'][i] = []
            self.recorder[ky]['num_joint_memory'][i] = []
        
        # record all the observations in the memory list
        self.recorder[ky]['num_y_joint_memory'][i].append(num_y_joint.detach().clone())
        self.recorder[ky]['num_joint_memory'][i].append(num_joint.detach().clone())

        # if the memory list length is higher than forget number, then starting forgetting the past observation.
        if len(self.recorder[ky]['num_joint_memory'][i]) > self.forget_num[ky]:
            self.recorder[ky]['num_y_joint'][i] = self.recorder[ky]['num_y_joint'][i] - self.recorder[ky]['num_y_joint_memory'][i].pop(0)
            self.recorder[ky]['num_joint'][i] = self.recorder[ky]['num_joint'][i] - self.recorder[ky]['num_joint_memory'][i].pop(0)
        
        if not i in self.recorder[ky]['num_joint']:
            self.recorder[ky]['num_y_joint'][i] = num_y_joint
            self.recorder[ky]['num_joint'][i] = num_joint
        else:
            self.recorder[ky]['num_y_joint'][i] = self.recorder[ky]['num_y_joint'][i] + num_y_joint
            self.recorder[ky]['num_joint'][i]  = self.recorder[ky]['num_joint'][i]  + num_joint

        # avoiding zeros at denominator
        self.recorder[ky]['num_joint'][i] = torch.clamp_min(self.recorder[ky]['num_joint'][i], self.stable_num[ky])
        # avoiding unstable connection between joint sample point and true labels.
        self.recorder[ky]['num_y_joint'][i] = torch.clamp_min(self.recorder[ky]['num_y_joint'][i], self.stable_num[ky])
        prob_y_joint= self.recorder[ky]['num_y_joint'][i] / self.recorder[ky]['num_joint'][i]

        # for very small value, due to system 'round' operation, sometimes the probability is higher than 1
        prob_y_joint = torch.clamp(prob_y_joint,0,1)


        return prob_y_joint

    def detach(self):
        ''' detach the following varaibles to avoid cycle graph during autograd of torch.'''
        
        for i in range(len(self.recorder['training']['num_y_joint'])):
            self.recorder['training']['num_y_joint'][i] = self.recorder['training']['num_y_joint'][i].detach()
            self.recorder['training']['num_joint'][i] = self.recorder['training']['num_joint'][i].detach()
            self.recorder['prediction']['num_y_joint'][i] = self.recorder['training']['num_y_joint'][i].detach()
            self.recorder['prediction']['num_joint'][i] = self.recorder['training']['num_joint'][i].detach()
        if self.recorder['mutual independent']['num_variables'] is not None:
            self.recorder['mutual independent']['num_variables'] = [_.detach() for _ in self.recorder['mutual independent']['num_variables']]
            self.recorder['mutual independent']['num_joint']  = self.recorder['mutual independent']['num_joint'].detach()


    def mutual_independence_loss(self, variables):
        ''' calculation of mutual independence of part/whole of random variable: A1, A2, ..., AN.
            Default configuration is set to be deactivated, this loss is not mandatory.
        '''
        ky = 'mutual independent'
        batch_size = variables[0].shape[0]
        # mutual independant
        num_variables = [torch.sum(_,dim = 0) for _ in variables]
        joint_variables = self.joint_sample_space(variables)
        num_joint = torch.sum(joint_variables,dim = 0)

        self.recorder[ky]['num_variables_memory'].append([_.detach().clone() for _ in num_variables])
        self.recorder[ky]['num_joint_memory'].append(num_joint.detach().clone())

        
        if len(self.recorder[ky]['num_variables_memory']) > self.forget_num[ky]:
            num_variables_old = self.recorder[ky]['num_variables_memory'] .pop(0)
            for k in range(len(num_variables_old)):
                self.recorder[ky]['num_variables'][k] = self.recorder[ky]['num_variables'][k] - num_variables_old[k]
            self.recorder[ky]['num_joint'] = self.recorder[ky]['num_joint']- self.recorder[ky]['num_joint_memory'].pop(0)
        

        if self.recorder[ky]['num_variables'] is None:
            self.recorder[ky]['num_variables'] = num_variables
            self.recorder[ky]['num_joint']  = num_joint
        else:
            for k in range(len(num_variables)):
                self.recorder[ky]['num_variables'][k] = self.recorder[ky]['num_variables'][k] + num_variables[k]
            self.recorder[ky]['num_joint']  = self.recorder[ky]['num_joint'] + num_joint
        number = len(self.recorder[ky]['num_variables_memory']) * batch_size
        prob_variables = [_/number for _ in self.recorder[ky]['num_variables']]
        prob_joint = self.recorder[ky]['num_joint'] / number
        prob_mul_variables = self.joint_sample_space(prob_variables,batched=False)
        
        prob_joint = torch.clamp_min(prob_joint,self.stable_num[ky])
        prob_mul_variables = torch.clamp_min(prob_mul_variables,self.stable_num[ky])
        loss = torch.sum(prob_joint*torch.log(prob_joint/prob_mul_variables))

        return loss
    
