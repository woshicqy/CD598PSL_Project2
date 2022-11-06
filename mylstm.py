import numpy as np
import pandas as pd
from datetime import date
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, ensemble

import warnings
warnings.filterwarnings("ignore")
class Optimizer:
    def __init__(self, lr=.1, beta_1=0.9, beta_2=0.999,
                 epsilon=0, decay=0., **kwargs):
        
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.iterations = 1
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.decay = decay
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_ADAM(self, params, grads):

        original_shapes = [x.shape for x in params]
        params = [x.flatten() for x in params]
        grads = [x.flatten() for x in grads]
        
        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * (np.sqrt(1. - np.power(self.beta_2, t)) /
                     (1. - np.power(self.beta_1, t)))

        if not hasattr(self, 'ms'):
            self.ms = [np.zeros(p.shape) for p in params]
            self.vs = [np.zeros(p.shape) for p in params]
    
        ret = [None] * len(params)
        for i, p, g, m, v in zip(range(len(params)), params, grads, self.ms, self.vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * np.square(g)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            self.ms[i] = m_t
            self.vs[i] = v_t
            ret[i] = p_t
        self.iterations += 1
  
        for i in range(len(ret)):
            ret[i] = ret[i].reshape(original_shapes[i])

        return np.array(ret)


    def get_SGD(self, w,p):
        for x,y in zip(w,p):
                    x+=self.lr*y
        return w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],w[9]

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values*(1-values)

def tanh_derivative(values): 
    return 1. - values ** 2

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args): 
    np.random.seed(0)
    return (np.random.rand(*args) * (b - a) + a)*.1

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim,optimization):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        
        self.opt=Optimizer()
        self.optimization=optimization

        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)

        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)


        
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 

    def apply_diff(self, lr = .1):
        if(self.optimization=='adam'):
            self.wg=self.opt.get_ADAM(self.wg,self.wg_diff)
            self.wi=self.opt.get_ADAM(np.array(self.wi),np.array(self.wi_diff))
            self.wf=self.opt.get_ADAM(np.array(self.wf),np.array(self.wf_diff))
            self.wo=self.opt.get_ADAM(np.array(self.wo),np.array(self.wo_diff))

        else:
            #This is the stochastic gradient descent code
            self.wg -= lr * self.wg_diff
            self.wi -= lr * self.wi_diff
            self.wf -= lr * self.wf_diff
            self.wo -= lr * self.wo_diff


        
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo) 
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo) 

class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
    
class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param

        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def bottom_data_is(self, x, s_prev = None, h_prev = None):
        # if this is the first lstm node in the network
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        # print(f'xc:{xc.shape}')
        # print(f'x:{x.shape}')
        # print(f'h_prev:{h_prev.shape}')
        # print(f'wg:{self.param.wg.shape}')
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc

    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di 
        df_input = sigmoid_derivative(self.state.f) * df 
        do_input = sigmoid_derivative(self.state.o) * do 
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        #for dparam in [self.param.wi_diff, self.param.wf_diff , self.param.wo_diff, self.param.wg_diff, self.param.bi_diff, self.param.bf_diff, self.param.bo_diff, self.param.bg_diff]:
        #    np.clip(dparam, -1, 1, out=dparam)

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]

class LstmNetwork():
    def __init__(self, lstm_param, loss):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence
        self.x_list = []
        self.loss=loss

    def y_list_is(self, y_list, loss_layer):

        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx],self.loss)

        diff_h =loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])

        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx],self.loss)
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1 

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
       # print(self.x_list)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)



class LossLayer:

    @classmethod
    def loss(self,pred, label,fn):
        if(fn=='mae'):
            return LossLayer.loss_mae(pred,label)
        else:
            return LossLayer.loss_rmse(pred,label)
    
    # MG added mean absolute error
    @classmethod
    def loss_mae(self, pred, label):
        return (np.abs(pred[0]-label))
        #return (pred[0] - label) ** 2
    
    @classmethod
    def loss_rmse(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] =2*(pred[0] - label)
        return diff



def modelTrain(X,Y,loss, optimization):
    mem_cell_ct = 50
    x_dim = 57
    lstm_param = LstmParam(mem_cell_ct, x_dim,optimization)
    lstm_net = LstmNetwork(lstm_param,loss)
    losses=[]
    bestLoss=1e5
       
    for ind in range(len(Y)):
        lstm_net.x_list_add(X[ind])

    loss = lstm_net.y_list_is(Y, LossLayer)
    losses.append(loss)

    if(loss<bestLoss):
        best_lstm_net = LstmNetwork(lstm_param,loss)
        
    lstm_param.apply_diff(lr=0.1)

    lstm_net.x_list_clear()
    
    # for ind in range(len(Y)):
    #     best_lstm_net.x_list_add(X[ind])   
    # loss = best_lstm_net.y_list_is(Y, LossLayer)
    return losses, [ lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(Y))],loss



def mypredict(train, test, next_fold, t):
    
    if t!=1:
        train = pd.concat([train,next_fold],ignore_index=True)

    # not all depts need prediction
    
    start_date = pd.to_datetime("2011-03-01") + relativedelta(months=2 * (t-1))
    end_date = pd.to_datetime("2011-05-01") + relativedelta(months=2 * (t-1))

    # find_week = lambda x : x.isocalendar()[1]+1  if x.isocalendar()[0] == 2010 else x.isocalendar()[1]

    find_week = lambda x : x.isocalendar()[1]
    find_yr = lambda x : x.isocalendar()[0]

    test['Wk'] =  pd.to_datetime(test['Date']).apply(find_week)
    train['Wk'] =  pd.to_datetime(train['Date']).apply(find_week)

    test['Yr'] =  pd.to_datetime(test['Date']).apply(find_yr)
    train['Yr'] =  pd.to_datetime(train['Date']).apply(find_yr)

    time_ids = (pd.to_datetime(test['Date'])>=start_date)&(pd.to_datetime(test['Date'])<end_date)
    test_current = test.loc[time_ids,]

    test_depts = test_current.Dept.unique()
    test_pred = None

    # print(train.head(5))
    trainY = train['Weekly_Sales']

    epochs = 50
    for epoch in range(epochs):

        avg_loss = []

        for dept in test_depts:    
        # no need to consider stores that do not need prediction
        # or do not have training samples
            train_dept_data = train[train['Dept']==dept]
            test_dept_data = test_current[test_current['Dept']==dept]
            train_stores = train_dept_data.Store.unique()
            test_stores = test_dept_data.Store.unique()
            test_stores = np.intersect1d(train_stores, test_stores)

            for store in test_stores:
                
                tmp_train = train_dept_data[train_dept_data['Store']==store]
                tmp_test = test_dept_data[test_dept_data['Store']==store]


                trainY = tmp_train['Weekly_Sales']
                tmp_train = tmp_train.drop(['Weekly_Sales'],axis=1)


                ohe = OneHotEncoder(handle_unknown='ignore',sparse=False,drop='if_binary')
                enc = ohe.fit_transform(tmp_train[['Wk','IsHoliday','Yr']])
        


                # ### encoding features ###
                train_dummy = pd.DataFrame(enc,columns=ohe.get_feature_names_out())
                test_dummy = pd.DataFrame(ohe.transform(tmp_test[['Wk','IsHoliday','Yr']]),columns=ohe.get_feature_names_out())

                train_dummy['Yr'] = tmp_train['Yr'].to_numpy()
                train_dummy['Store'] = tmp_train['Store'].to_numpy()
                train_dummy['Dept'] = tmp_train['Dept'].to_numpy()

                test_dummy['Yr'] = tmp_test['Yr'].to_numpy()
                test_dummy['Store'] = tmp_test['Store'].to_numpy()
                test_dummy['Dept'] = tmp_test['Dept'].to_numpy()

                train_dummy_array = train_dummy.to_numpy()
                trainY_array = trainY.to_numpy()


                test_dummy_array = test_dummy.to_numpy()

                losses, predictions,loss = modelTrain(train_dummy_array,trainY_array,'mae','sgd')
                avg_loss += losses
        
        if(epoch%5==0):
            print("iter", "%2s" % str(epoch), end=": ")
        
        averageLoss = np.mean(avg_loss)
        if(epoch%50==0):
            print("loss:", "%.3e" % averageLoss)
    # print('train shape:',train_dummy.shape)
    # print('test shape:',test_dummy.shape)

    # print('test shape:',train_dummy_array.shape)
    # print('test shape:',test_dummy_array.shape)
    # exit()

    # ### covert to dataframe ###


    
    return train,test_pred



if __name__ == '__main__':

    train = pd.read_csv('train_ini.csv', parse_dates=['Date'])
    test = pd.read_csv('test.csv', parse_dates=['Date'])

    # save weighed mean absolute error WMAE
    n_folds = 10
    next_fold = None
    wae = []

    for t in range(1, n_folds+1):
        print(f'Fold{t}...')

        # *** THIS IS YOUR PREDICTION FUNCTION ***
        train, test_pred = mypredict(train, test, next_fold, t)

        # Load fold file
        # You should add this to your training data in the next call to mypredict()
        fold_file = 'fold_{t}.csv'.format(t=t)
        next_fold = pd.read_csv(fold_file, parse_dates=['Date'])

        # extract predictions matching up to the current fold
        scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

        # extract weights and convert to numpy arrays for wae calculation
        weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
        actuals = scoring_df['Weekly_Sales'].to_numpy()
        preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

        wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

    print('WAE:',wae)
    print(sum(wae)/len(wae))