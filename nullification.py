import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt  
import numpy as np
import pickle

from LRP import LayerwiseRelevancePropagation
from utils import *

def get_sorted_relevannce_index(relevance):
    relevance=np.reshape(relevance, (120,))
    sort_index = np.argsort(relevance)

    return sort_index

def Nullification( dataset, model, relevance, target, order):
    Test_X, Test_Y = dataset
    sample_index= get_same_class_index(target, dataset) #200
    correct_index= get_corr_pred_index(target, dataset, model) #190
    incorrect_index= get_incorr_pred_index(target, dataset, model) #10
    result=[]
    sorted_r_index=[] #102x120
    x1=[] 
    y1=[] 
    x2=[] 
    y2=[] 

    for index in incorrect_index:#200
        x2.append(Test_X[index])
        y2.append(Test_Y[index])

    for index in correct_index: #104
        r= relevance[index]
        r_index= get_sorted_relevannce_index(r)
        if order == 1:
            r_index = r_index[::-1]
        sorted_r_index.append(r_index) #104
        x1.append(Test_X[index])
        y1.append(Test_Y[index])

    x=x1+x2
    y=y1+y2
    x=np.asarray(x) 
    y=np.asarray(y)

    score= model.evaluate(x,y,verbose=0)
    result.append(score[1])

    sorted_r_index= np.asarray(sorted_r_index)
    sorted_r_indext=np.transpose(sorted_r_index)
    # start nullification
    for i in range(sorted_r_index.shape[1]):
        for j in range(sorted_r_index.shape[0]):

            x[j][sorted_r_indext[i][j]]=0
        #print(model.predict_classes(x))
        score= model.evaluate(x,y,verbose=0)
        result.append(score[1])

    return result

def Visulization(target, descending_result, ascending_result):
    '''
    Show CH-NULLIFICATION on target label 
    '''
    plt.plot( descending_result,'b-', label='Descending')
    plt.plot( ascending_result,'g--', label='Ascending')
    plt.title('Location $p_{%d}$'%(target+1) , fontsize=20)
    plt.legend(loc='best',fontsize=18)
    plt.axis([0, 120, 0, 1])
    plt.xlabel('Number of Nullified Channels',fontsize=20)
    plt.ylabel('Recall',fontsize=20)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.show()
    #plt.savefig('CN-p%d.png'%(target+1), dpi=400)
    #plt.savefig('CN-p%d.png'%(target+1).eps'%(target+1), format='eps')
    #plt.close()



if __name__ == '__main__':
    # load keras model
    model_name = 'models/32CNN.h5'
    model = load_model(model_name)
    model.summary()
    # load CSI samples
    Test_X = np.array(pickle.load(open('CSI/exp1/Test_X.pickle','rb')))
    Test_Y = np.array(pickle.load(open('CSI/exp1/Test_y.pickle','rb')))
    Test_X = Test_X/100.0 #scale CSI
    Test_X =Test_X.reshape(-1,120,1)
    #Test_X = np.reshape(Test_X, (Test_X.shape[0], Test_X.shape[1],1))

    # compute relevance score
    relevance = LayerwiseRelevancePropagation(Test_X, Test_Y, model).run_lrp()

    # start channel nullification
    target = 0 # location 1

    order = 1 # Descending
    descending_result = Nullification( (Test_X, Test_Y), model, relevance, target, order)
    order = 0 # Ascending
    ascending_result = Nullification( (Test_X, Test_Y), model, relevance, target, order)

    Visulization(target, descending_result, ascending_result)