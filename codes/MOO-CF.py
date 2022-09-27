from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from tensorflow import keras
import argparse
import shutil
import tensorflow as tf
from keras.models import load_model
import os
sys.path.append('../')
import copy
from keras import backend as K

from keras.applications.vgg16 import preprocess_input

import random
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


def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    #print("tempshape:",temp.shape)
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

def Select_seeds(seedMap):   
    '''
    input: seedMap: a numpy of (n*m) {0/1}
    output: resSeed: a list of selected seeds
    '''
    
    resSeed=[]
    seed_map=copy.deepcopy(seedMap)
    x=0
    Variable=[]
    for i in range(seed_map.shape[0]):
            if i==0:
                Variable=sum(seed_map[i])
            else:
                Variable=np.append(Variable,sum(seed_map[i])) 
    while sum(Variable)>0:
        Variable=[]
        for i in range(seed_map.shape[0]):
            if i==0:
                Variable=sum(seed_map[i])
            else:
                Variable=np.append(Variable,sum(seed_map[i])) 
        ranks = np.argsort(-Variable)
        resSeed.append(ranks[0])
        for i in range(seed_map.shape[1]):
            if seed_map[ranks[0],i]==1:
                seed_map[:,i]=0
        x=x+1
        #print(x)
    return resSeed

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

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(1, 32, 32, 3)
    temp = temp.astype('float32')
    temp /= 255
    return temp

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

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory',default='./mnist_all')
    parser.add_argument('-coverage', help='coverage information directory',default='./mnist_kmnc')
    parser.add_argument('-model', help="target model to fuzz", choices=['resnet20', 'lenet5','lenet5_fm','svhnnet'], default='lenet5')
    parser.add_argument('-seednum', help="the expected number of seeds", type=int, default=200)
    parser.add_argument('-out', help='output directory')
    args = parser.parse_known_args()[0]

    model = load_model(model_weight_path[args.model])
    #model = keras.models.load_model(model_weight_path[args.model])

    # Get the preprocess function based on different dataset
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
    print(stage.shape)
    coverage_seed = Select_seeds(stage)
    list=[]
    pcs=[]
    i=0
    for seed_name in seed_lis:
        if not os.path.isdir(seed_name):
            s = np.load(file+"/"+seed_name)
            eachpcs=PCS(preprocess(s),model)
            for j in coverage_seed:
                if seed_name == tasks[j]:
                    eachpcs=2
                    list=np.append(list,i)
            if i==0:
                pcs=eachpcs
                i=i+1
            else:
                pcs=np.append(pcs,eachpcs) 
                i=i+1
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
    for i in list:
        fols[int(i)]=-10000
    grad_rank=rank(-fols)
    
    com=np.stack(((pcs_rank),(grad_rank)),axis=1)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = com)
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    else:
        shutil.rmtree(args.out)
        os.makedirs(args.out)
    if args.seednum < len(coverage_seed):
        coverage_seed1=random.sample(coverage_seed,int(args.seednum/2)) #You can change the number of seeds selected by coverage-guided strategy to what you want. 
        for seed_name in coverage_seed1:
            fn="%s/%s" % (args.out, str(seed_name))
            np.save(fn,np.load(file+"/"+str(seed_name)+".npy")) 
    else:
        for seed_name in coverage_seed:
            fn="%s/%s" % (args.out, str(seed_name))
            np.save(fn,np.load(file+"/"+str(seed_name)+".npy"))
    print('coverage-selected seed number:',len(os.listdir(args.out)))

    num=len(os.listdir(args.out))    
    for i in range(len(ndf)):
        if len(ndf[i])<= (args.seednum-num):  
            for j in ndf[i]:
                fn="%s/%s" % (args.out, (str(j)+".npy"))
                np.save(fn,np.load(file+"/"+str(j)+".npy"))
        else:
            neednum=args.seednum-num    
            index=random.sample(ndf[i].tolist(),neednum)
            for m in index:
                fn="%s/%s" % (args.out, (str(m)+".npy"))
                np.save(fn,np.load(file+"/"+str(m)+".npy"))       
        num=len(os.listdir(args.out))
    print('total seed number:',len(os.listdir(args.out)))