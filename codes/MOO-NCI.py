from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from tensorflow import keras
import argparse, pickle
import shutil
import tensorflow as tf
from keras.models import load_model
import os
from keras import Input
sys.path.append('../')
import copy
from keras import backend as K

import random
import time
import numpy as np
import pygmo as pg
from tensorflow.keras.datasets import cifar10,mnist,fashion_mnist
tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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

def coverage(seedMap,n):   
    '''
    input: seedMap: a numpy of (n*m) {0/1}
    output: y: a list of seed rank using coverage sorting
    '''
    
    resSeed=[]
    seed_map=copy.deepcopy(seedMap)
    y=np.zeros((1,seedMap.shape[0]))
    y=y.reshape(y.shape[1],)
    x=0
    while x<n:
        Variable=[]
        for i in range(seed_map.shape[0]):
            if i==0:
                Variable=sum(seed_map[i])
            else:
                Variable=np.append(Variable,sum(seed_map[i])) 
        if sum(Variable)==0:
            Variable=[]
            seed_map=copy.deepcopy(seedMap)
            for j in resSeed:
                seed_map[j,:]=0
            for i in range(seed_map.shape[0]):
                if i==0:
                    Variable=sum(seed_map[i])
                else:
                    Variable=np.append(Variable,sum(seed_map[i])) 
        ranks = np.argsort(-Variable) 
        resSeed.append(ranks[0])
        y[ranks[0]]=x
        for i in range(seed_map.shape[1]):
            if seed_map[ranks[0],i]==1:
                seed_map[:,i]=0        
        x=x+1
        print('coverage ranking:',x,'/',seedMap.shape[0])
    return y

def PCS(x,M):
    model=M
    y = model.predict(x)
    y=y.reshape(y.shape[1],)
    y=np.sort(y)
    pcs=y[-1]-y[-2]
    return pcs

def rank(x):
    y=np.zeros((x.shape[0],))
    ranks = np.argsort(x)
    for i in range(len(x)):
        y[ranks[i]]=i
    return y

model_weight_path = {
    'resnet20': "./Seed-Selection/models/cifar-10/resnet20.h5",
    'lenet5': "./Seed-Selection/models/mnist/lenet5.h5",
    'lenet5_fm':'./Seed-Selection/models/fashion-mnist/lenet5.h5',
    'svhnnet':'./Seed-Selection/models/svhn/cnn-model.h5'
}

preprocess_dic = {
    'resnet20': cifar_preprocessing,
    'lenet5': mnist_preprocessing,
    'lenet5_fm': mnist_preprocessing,
    'svhnnet': svhn_preprocessing
}


        
if __name__ == '__main__':

    start_time = time.time()
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='MOO(NCI)')

    parser.add_argument('-i', help='input seed directory',default='./mnist_all')
    parser.add_argument('-coverage', help='coverage information directory',default='./mnist_kmnc')
    parser.add_argument('-model', help="target model to fuzz", choices=['resnet20', 'lenet5','lenet5_fm','svhnnet'], default='lenet5')
    parser.add_argument('-seednum', help="the expected number of seeds", type=int, default=500)
    parser.add_argument('-out', help='output directory')

    args = parser.parse_args()

    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)
    
    model = load_model(model_weight_path[args.model])
    #model = keras.models.load_model(model_weight_path[args.model])
    preprocess = preprocess_dic[args.model]
    file=args.i
    seed_lis=os.listdir(file)
    seed_lis.sort(key= lambda x: int(x[:-4])) 
    file_dir=args.coverage
    
    tasks=os.listdir(file_dir)
    tasks.sort(key= lambda x: int(x[:-4])) 

    stage=[[]]
    i=0
    for c_n in tasks:
        if not os.path.isdir(c_n): 
            f = np.load(file_dir+"/"+c_n)
            n_num=f.shape[0]
            f=f.reshape(1,n_num)
            if i==0:
                stage=f
                i=i+1
            else:
                stage=np.append(stage,f,axis=0)
    coverage_rank=coverage(stage,stage.shape[0])
    
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
                pcs=np.append(pcs,eachpcs) 
    pcs_rank=rank(pcs)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test1 = x_test/255
    x_test1 = x_test1.reshape(x_test.shape[0], 28, 28, 1)
    #x_test1 = x_test1.reshape(x_test.shape[0], 32, 32, 3)
    y_test = keras.utils.to_categorical(y_test, 10)
    gen_img = tf.Variable(x_test1)
    with tf.GradientTape() as g:
        loss = keras.losses.categorical_crossentropy(y_test, model(gen_img))
        grads = g.gradient(loss, gen_img)
    fols = np.linalg.norm((K.eval(grads)+1e-20).reshape(x_test1.shape[0], -1), ord=2, axis=1)
    grad_rank=rank(-fols)
    
    com=np.stack(((coverage_rank),(pcs_rank),(grad_rank)),axis=1)
    value=coverage_rank+pcs_rank+grad_rank
        
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = com)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    else:
        shutil.rmtree(args.out)
        os.makedirs(args.out)
    num=len(os.listdir(args.out))    
    for i in range(len(ndf)):
            if len(ndf[i])<= (args.seednum-num):  
                for j in ndf[i]:
                    fn="%s/%s" % (args.out, (str(j)+".npy"))
                    np.save(fn,np.load(file+"/"+str(j)+".npy"))   
            else:
                neednum=args.seednum-num
                value_list=[]        
                for j in ndf[i]:    
                    value_list=np.append(value_list,value[j])
                ranks = np.argsort(value_list)
                index=ranks[:neednum]
                index=random.sample(list(ndf[i]),neednum)
                for m in index:
                    fn="%s/%s" % (args.out, (str(m)+".npy"))
                    np.save(fn,np.load(file+"/"+str(m)+".npy")) 
            num=len(os.listdir(args.out))
    print('seed number:', len(os.listdir(args.out)))
