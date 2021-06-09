from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt  
import numpy as np
import pickle

from LRP import LayerwiseRelevancePropagation
from utils import *

def compute_heatmaps(index):

    x = list(range(1,121))
    print("Numbers of superimposed CSI samples: ", len(index))
    for i in index:
        plt.scatter(x, X[i]*100, c=relevance[i], s=2, cmap='Reds', vmin=0, vmax=1)

    plt.title('Location $p_{%d}$'%(label+1) , fontsize=20)
    plt.xlabel('Channel Index', fontsize=18)
    plt.ylabel('CSI Amplitude', fontsize=18)

    #plt.title('Location $p_{%d}$, %s, Run:0, Recall:%.2f'%(label+1,name,score) , fontsize=15)
    cbar= plt.colorbar()
    cbar.ax.set_ylabel('Relevance Scores', rotation=270, fontsize=14,labelpad=20)
    plt.axis([0, 120, 0, 40])
    plt.grid(True)  
    plt.show()
    #plt.savefig('C:/Users/acicula/Desktop/expresult/Final/LRP3/[heatmap][%s][p%d].png'%(self.type, label+1), dpi=400)
    

if __name__ == '__main__':
    model_name = 'models/32CNN.h5'
    model = load_model(model_name)
    model.summary()
    X = np.array(pickle.load(open('CSI/exp1/Test_X.pickle', 'rb')))
    X = X.reshape(-1, 120, 1)
    X = X / 100.0
    Y = np.array(pickle.load(open('CSI/exp1/Test_y.pickle', 'rb')))
    relevance = LayerwiseRelevancePropagation(X, Y, model).run_lrp()
    relevance = minmaxscale(relevance, (0,1)) #normalize 
    label = 0

    index = get_same_pred_index(label, [X, Y], model)
    compute_heatmaps(index)
    #CX=np.reshape(X, (X.shape[0], X.shape[1],1))
    #cnnplot=output_heatmap(CX,Y,'32CNN.h5')
    ###########################initialization###############################




