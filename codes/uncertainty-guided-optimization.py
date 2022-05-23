from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from keras import backend as K

import argparse
import shutil

from keras.models import load_model
import tensorflow as tf
import os
sys.path.append('../')
import random
import numpy as np
import math

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(1, 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(1,32, 32, 3)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(1, 32, 32, 3)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def PCS(x,M):
    model=M
    y = model.predict(x)
    y=y.reshape(y.shape[1],)
    y=np.sort(y)
    pcs=y[-1]-y[-2]
    return pcs

def gini( x,model):
    """
    Different from the defination in DeepGini paper (deepgini = 1 - ginis), the smaller the ginis here, the larger the uncertainty. 
    shape of x: [batch_size, width, height, channel]
    """
    x = tf.Variable(x)
    preds = K.eval(model(x))
    ginis = np.sum(np.square(preds), axis=1)
    return ginis

def select(pcs, n, s='low'):
    if s == 'KDV':
        index1,index2,index3,index4,index5,index6,index7,index8,index9,index10,index =[list() for x in range(11)]
        k=math.ceil(n/10)
        for i in range(len(pcs)):
            if pcs[i]>=0.9: index10.append(i)
            elif 0.8<=pcs[i]<0.9: index9.append(i)
            elif 0.7<=pcs[i]<0.8: index8.append(i)
            elif 0.6<=pcs[i]<0.7: index7.append(i)
            elif 0.5<=pcs[i]<0.6: index6.append(i)
            elif 0.4<=pcs[i]<0.5: index5.append(i)
            elif 0.3<=pcs[i]<0.4: index4.append(i)
            elif 0.2<=pcs[i]<0.3: index3.append(i)
            elif 0.1<=pcs[i]<0.2: index2.append(i)
            elif pcs[i]<0.1: index1.append(i)    
        indexes=[index1,index2,index3,index4,index5,index6,index7,index8,index9,index10]
        for ind in indexes:
            print(len(ind))
            if len(index)>(n-k): indexsingle = random.sample(list(ind), (n-len(index)))
            else: indexsingle = random.sample(list(ind), k)
            index.extend(indexsingle)
        return index

    elif s == 'high':
        ranks = np.argsort(-pcs)
        return ranks[:n]
    elif s == 'low':
        ranks = np.argsort(pcs)
        return ranks[:n]  

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uncertainty guided seed selection')

    parser.add_argument('-seed', help='seed corpus directory')
    parser.add_argument('-seednum', help="expected number of seeds",default='100')
    parser.add_argument('-model', help='model path')
    parser.add_argument('-out', help='output directory')

    args = parser.parse_args()
    
    model = load_model(args.model)

    preprocess = mnist_preprocessing   #change if use other dataset

    file=args.seed
    seed_lis=os.listdir(file)
    seed_lis.sort(key= lambda x: int(x[:-4]))
    
    pcs=[]
    i=0
    for seed_name in seed_lis:
        if not os.path.isdir(seed_name):
            s = np.load(file+"/"+seed_name)
            eachpcs=PCS(preprocess(s),model)
            if i==0:
                pcs=eachpcs
                i=i+1
            else:
                pcs=np.append(pcs,eachpcs) #array of PCS

    index=select(pcs, n=args.seednum, s='low')

    if not os.path.exists(args.out):
        os.makedirs(args.out) 
    else:
        shutil.rmtree(args.out)
        os.makedirs(args.out)
    for j in index:
        fn="%s/%s" % (args.out, str(j))
        np.save(fn,np.load(file+"/"+str(j)+".npy"))
    print(len(os.listdir(args.out)))
    
    
