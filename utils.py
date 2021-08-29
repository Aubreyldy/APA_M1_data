#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time, random
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold


# Causalml
import causalml

from causalml.dataset import synthetic_data
from causalml.metrics.visualize import *
from causalml.propensity import calibrate

import logging
logger = logging.getLogger('causalml')
logger.setLevel(logging.DEBUG)
plt.style.use('fivethirtyeight')

# Benchmark
from xgboost import XGBRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor, LRSRegressor
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, TMLELearner

# Dragonnet
from causalml.inference.nn import DragonNet

# Causal net
import causal_nets
from causal_nets import causal_net_estimate

# Tuning
#from bayes_opt import BayesianOptimization #Cannot work
from sklearn.model_selection import GridSearchCV




# Sensitivity Analysis Function
def cv_cn(method, X_train, X_test, T_train, T_test, Y_train, Y_test, params):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # This cross-validation object is a variation of KFold that returns
    # stratified folds. The folds are made by preserving the percentage of
    # samples for each class.

    X_train = X_train.values
    Y_train = Y_train.values
    T_train = T_train.values

    x_test = X_test.values
    y_test = Y_test.values
    t_test = T_test.values

    auuc = []
    Tau_pred = []
    qini = []
    if method == "Causal_nn":
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, Y_train)):
            # Seperate the training data into training and validation subsets
            x_train, x_val = X_train[train_index], X_train[val_index]
            y_train, y_val = Y_train[train_index], Y_train[val_index]
            t_train, t_val = T_train[train_index], T_train[val_index]

            # Only need the estimated CATE
            tau_pred, _, _, _, _, _, _ = causal_net_estimate(
                [x_train, t_train, y_train], [x_val, t_val, y_val], [x_test, t_test, y_test],
                hidden_layer_sizes=params['1'], dropout_rates=params['2'], batch_size=None, alpha=0.,
                r_par=0., optimizer='Adam', learning_rate=params['3'],
                max_epochs_without_change=30, max_nepochs=10000, seed=123, estimate_ps=False)
            Tau_pred.append(tau_pred.mean())


    elif method == "Dragon_n":

        # Try to fix the result of DragonNet for each time, but it seems to fail.
        # But actually it does not matter a lot
        rnd_state = 888
        random.seed(a=rnd_state)

        # The inbuilt function cannot be fixed by random seed
        dragon = DragonNet(neurons_per_layer=params['1'], targeted_reg=True,
                           ratio=params['2'], learning_rate=params['3'])

        # Fit the model via traning set
        dragon.fit(X_train, T_train, Y_train)

        # Predict the model via test set
        dragon_ite = dragon.predict(x_test, t_test, y_test)

        # Calculate the ITE
        tau_pred = (dragon_ite[:, 1] - dragon_ite[:, 0])

        # Calculate the CATE
        Tau_pred.append(tau_pred.mean())

    elif method == "T_learner":
        rnd_state = 888
        random.seed(a=rnd_state)
        learner_t = BaseTRegressor(learner=XGBRegressor(learning_rate=params['1']))
        learner_t.fit(X=X_train, treatment=np.ravel(T_train), y=np.ravel(Y_train))
        tau_pred = learner_t.predict(X=x_test, treatment=np.ravel(t_test), y=np.ravel(y_test)).flatten()
        Tau_pred.append(tau_pred.mean())

    elif method == "S_learner":
        rnd_state = 888
        random.seed(a=rnd_state)
        learner_s = BaseSRegressor(learner=XGBRegressor(learning_rate=params['1']))
        learner_s.fit(X=X_train, treatment=np.ravel(T_train), y=np.ravel(Y_train))
        tau_pred = learner_s.predict(X=x_test, treatment=np.ravel(t_test), y=np.ravel(y_test)).flatten()
        Tau_pred.append(tau_pred.mean())

    df = pd.DataFrame({'y': np.ravel(y_test), 'w': np.ravel(t_test), 'Causal-net': tau_pred.flatten()})
    auuc.append(auuc_score(df, outcome_col='y', treatment_col='w')[0])
    qini.append(qini_score(df, outcome_col='y', treatment_col='w')[0])

    return np.mean(Tau_pred), np.mean(auuc), np.mean(qini)


# Plot Function

def ourplot(xaxis, file1, file2, y11, y12, y21, y22, x, title, num, nn=0, lab=0):
    plt.figure(figsize=(10, 4))
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.plot(xaxis, file1[y11], c='red')
    plt.xlabel(x)
    plt.ylabel('ATE')
    if num != 1:
        plt.plot(xaxis, list(file2[y12]) * nn, label=lab)
        plt.legend(loc='lower center')

    plt.subplot(1, 2, 2)
    plt.plot(xaxis, file1[y21], c='blue')
    plt.xlabel(x)
    plt.ylabel('AUUC')
    if num != 1:
        plt.plot(xaxis, list(file2[y22]) * nn, label=lab)
        plt.legend(loc='lower center')

    plt.tight_layout()
    plt.suptitle(title, fontsize=14)
    plt.show()

    return plt


'''# Tune Hidden Layer Size
ATE_hls = []
AUUC_hls = []
QINI_hls = []

for i in range(10,41):
    Params = {'1': ([30,i]),
        '2': ([0.5,0]),
        '3': 0.0001}
    ate_hls, auuc_hls, qini_hls= cv_cn("Causal_nn",X_train_c, X_test_c,
                                       T_train_c, T_test_c, Y_train_c, Y_test_c,Params)
    ATE_hls.append(ate_hls)
    AUUC_hls.append(auuc_hls)
    QINI_hls.append(qini_hls)

ATE_AUUC_hls = pd.DataFrame([ATE_hls,AUUC_hls, QINI_hls]).transpose()
ATE_AUUC_hls.columns = ['ATE_hls','AUUC_hls','QINI_hls']
ATE_AUUC_hls.to_csv('/Users/aubrey/Desktop/HU Berlin/APA/ATE_AUUC_hls.csv')'''



'''# Dropout Rate
ATE_dr = []
AUUC_dr = []
QINI_dr = []

for i in np.linspace(0,0.9,num=10):
    Params = {'1': ([30,18]),
        '2': ([i,0]),
        '3': 0.0001}
    ate_dr, auuc_dr, qini_dr= cv_cn("Causal_nn",X_train, X_test, T_train, T_test, Y_train, Y_test,Params)
    ATE_dr.append(ate_dr)
    AUUC_dr.append(auuc_dr)
    QINI_dr.append(qini_dr)

ATE_AUUC_dr = pd.DataFrame([ATE_dr,AUUC_dr, QINI_dr]).transpose()
ATE_AUUC_dr.columns = ['ATE_dr','AUUC_dr', 'QINI_dr']

ATE_AUUC_dr.to_csv('/Users/aubrey/Desktop/HU Berlin/APA/ATE_AUUC_dr.csv')'''

'''# Delete the second layer
ATE_hln = []
AUUC_hln = []
QINI_hln = []

Params = {'1': [30],
        '2': [0.5],
        '3': 0.0001}
ate_hln, auuc_hln, qini_hln= cv_cn("Causal_nn",X_train, X_test, T_train, T_test, Y_train, Y_test,Params)
ATE_hln.append(ate_hln)
AUUC_hln.append(auuc_hln)
QINI_hln.append(qini_hln)

ATE_AUUC_hln = pd.DataFrame([ATE_hln,AUUC_hln, QINI_hln]).transpose()
ATE_AUUC_hln.columns = ['ATE_hln','AUUC_hln', 'QINI_hln']
ATE_AUUC_hln.to_csv('/Users/aubrey/Desktop/HU Berlin/APA/ATE_AUUC_hln.csv')
'''

'''# Meta-parameter of one-layer-nn
ATE_AUUC_2 = pd.DataFrame(columns=['hln_2','dr_2','lr_2','ATE_2','AUUC_2'])
# index for store
s=1
for i in range(30,82,20):
    for j in np.linspace(0.5,0.8,num=4):
        for t in np.linspace(0.0001,0.001,num=4):
            Params = {'1': [i],
                    '2': [j],
                    '3': t}
            ate_2, auuc_2= cv_cn("Causal_nn",X_train, X_test, T_train, T_test, Y_train, Y_test,Params)
            ATE_AUUC_2.loc[s, ['hln_2']] = i
            ATE_AUUC_2.loc[s, ['dr_2']] = j
            ATE_AUUC_2.loc[s, ['lr_2']] = t
            ATE_AUUC_2.loc[s, ['ATE_2']] = ate_2
            ATE_AUUC_2.loc[s, ['AUUC_2']] = auuc_2
            s=s+1

ATE_AUUC_2.to_csv('/Users/aubrey/Desktop/HU Berlin/APA/ATE_AUUC_2.csv') 
'''

'''# Meta-parameter of one-layer-nn
start_time = time.time()

#dragon1 = pd.DataFrame(columns=['nrp','r','lr','ate','auuc'])
# index for store
s=1
for i in range(170,230,10):
    for j in np.linspace(0.4,0.8,num=5):
        for h in np.linspace(0.0002,0.001,num=5):
            Params = {'1': i,
                    '2': j,
                    '3': h}
            ate_1, auuc_1, qini_1= cv_cn("Dragon_n",X_train_c, X_test_c,
                                    T_train_c, T_test_c, Y_train_c, Y_test_c,Params)
            dragon1.loc[s, ['nrp']] = i
            dragon1.loc[s, ['r']] = j
            dragon1.loc[s, ['lr']] = h
            dragon1.loc[s, ['ate']] = ate_1
            dragon1.loc[s, ['auuc']] = auuc_1
            dragon1.loc[s, ['qini']] = qini_1
            s=s+1

print("--- %s seconds ---" % (time.time() - start_time))

dragon1.to_csv('/Users/aubrey/Desktop/HU Berlin/APA/dragon1.csv') 
'''