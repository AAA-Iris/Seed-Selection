
#from __future__ import print_function
import numpy as np
import keras
import os
import argparse
from keras.datasets import cifar10, mnist,fashion_mnist
from keras.utils import np_utils
from keras.models import load_model
import random
from keras import backend as K
from tensorflow import keras
import tensorflow as tf

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp
def mnist_preprocessing1(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(-1, 28, 28, 1)
    temp = temp.astype('float32')
    return temp
def cifar_preprocessing1(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(-1, 32, 32, 3)
    temp = temp.astype('float32')
    return temp

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 32, 32, 3)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    #temp=temp.reshape(1,32,32,3)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def PCS(x,M):
    model=M
    y = model.predict(x)
    y=y.reshape(y.shape[1],)
    y=np.sort(y)
    pcs=y[-1]-y[-2]
    return pcs

def get_initialx(paths,initialseedpath):
    i=0
    for c_n in os.listdir(paths): 
        path=c_n[-10:-5]
        path=path.lstrip("0")
        if len(path)==0: path='0'   
        f = np.load(initialseedpath+"/"+path+'.npy')
        f=preprocess1(f)  
        if i==0:  
            x=f
            i=i+1    
        else:
            x = np.concatenate((x, f))
    return x

def FOL( x,xi, y, model):
    ep=0.3
    fols = []
    with tf.device('/gpu:1'):        
        target = tf.constant(y)
         
    with tf.device('/gpu:1'):        
        x = tf.Variable(x)
        xi = tf.Variable(xi)    
    x_adv,xi=x, xi
        
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(target, model(x_adv))
        grads = tape.gradient(loss, x_adv)
        grad_norm = np.linalg.norm(K.eval(grads).reshape(K.eval(x_adv).shape[0], -1), ord=1, axis=1)
        
        grads_flat = K.eval(grads).reshape(K.eval(x_adv).shape[0], -1)
        diff = (K.eval(x_adv) - K.eval(xi)).reshape(x_adv.shape[0], -1)
        for i in range(x_adv.shape[0]):
            i_fol = -np.dot(grads_flat[i], diff[i]) + ep * grad_norm[i]
            fols.append(i_fol)   
    return np.array(fols)

def select(foscs, n, ss, k=10):
    """
    n: the number of selected test cases. 
    s: strategy, ['best', 'kmst', 'high']
    k: for KM-ST, the number of ranges. 
    """
    ranks = np.argsort(foscs)
    # we choose test cases with small and large fols. 
    if ss == 'best':
        h = n//2
        return np.concatenate((ranks[:h],ranks[-h:]))

    # we choose test cases with small and large fols.   
    elif ss == 'kmst':
        index = []
        section_w = len(ranks) // k
        section_nums = n // section_w
        indexes = random.sample(list(range(k)), section_nums)
        for i in indexes:
            block = ranks[i*section_w: (i+1)*section_w]
            index.append(block)
        return np.concatenate(np.array(index))
    
    elif ss == 'high':
        return ranks[-n:] 

    else:
        return ranks[:n] 

def PCS_select(seed_numpy,n,model,ss):
    pcs=[]
    i=0
    for s in seed_numpy:
        s=s.reshape(1,s.shape[0],s.shape[1],s.shape[2])
        eachpcs=PCS(preprocess(s),model)
        if i==0:
            pcs=eachpcs
            i=i+1
        else:
            pcs=np.append(pcs,eachpcs) #pcs array
    print(pcs.shape)
    indexes=select(pcs,n,ss)
    return indexes[:n] 

def get_list(paths):
    crash=[]
    for c_n in os.listdir(paths):    
        if not os.path.isdir(c_n):  
            f = np.load(paths+"/"+c_n)
            crash.append(f)
    return crash

def get_numpy(paths):
    i=0
    for c_n in os.listdir(paths): 
        f = np.load(paths+"/"+c_n)
        f=preprocess1(f)  
        if i==0:  
            x=f
            i=i+1    
        else:
            x = np.concatenate((x, f),axis=0)
    return x

def get_ynumpy(paths):
    i=0
    for c_n in os.listdir(paths): 
        f = np.load(paths+"/"+c_n)
        f=np_utils.to_categorical(f, 10) 
        f=f.reshape(-1,10)  
        if i==0:  
            x=f
            i=i+1    
        else:
            x = np.concatenate((x, f),axis=0)
    return x

def select_function(crash_images, ground_truth, xi_path,x_numpy,y_numpy,select_ratio, dataset,s,ss,set_select_number=None, random_seed=0):
    """
    :param crash_images: test error
    :param ground_truth: correct label of test error
    :param select_ratio: the proportion of samples that are selected from test errors to put into the original dataset in the total number of test errors
    :param set_select_number: the number of samples that are selected from test errors to put into the original dataset, if we set this parameters, the select_ratio would not be used.
    :return X_retrain_shuffled: shuffled retrain dataset
    :return y_retrain_shuffled: shuffled label of retrain dataset
    :return remain_images: remaining test errors
    :return remain_truths: label of remaining test errors
    """

    if dataset=='mnist':
        model = load_model("./mnist/models/lenet5.h5")  #the path of model
        initialseedpath='./mnist_all'   #the path of initial seeds, contains several files which preserve single array format seed in each file
    elif dataset == 'cifar':
        model = keras.models.load_model("./cifar10/models/resnet.h5")
        initialseedpath='./cifar_all'
    elif dataset == 'fashionmnist':
        model = keras.models.load_model("./fashion-mnist/models/lenet5.h5")
        initialseedpath='./fashionmnist_all'
    elif dataset == 'svhn':
        model = keras.models.load_model("./SVHN/cnn_model.h5")
        initialseedpath='./svhn_all'
    image_length = len(crash_images)
    if set_select_number == None:
        select_number = round(image_length * select_ratio)
    else:
        select_number = set_select_number
    
    if s=='random':
        select_index = np.random.choice(range(image_length), select_number, replace=False)
    elif s == 'pcs':
        select_index=PCS_select(crash_images,select_number,model,ss)

    elif s == 'fol':
        xi=get_initialx(xi_path,initialseedpath)
        fol=FOL(x_numpy,xi, y_numpy, model,randRate=1)
        select_index=select(fol,select_number,model,ss)

    select_images = np.array(crash_images)[select_index]
    select_truths = np.array(ground_truth)[select_index]

    # the remaining images can be used to be the validation dataset
    remain_index = np.arange(image_length)
    remain_index = np.delete(remain_index, select_index)
    remain_images = np.array(crash_images)[remain_index]
    remain_truths = np.array(ground_truth)[remain_index]

    # preprocess
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_retrain = np.concatenate((X_train, select_images))
        y_retrain = np.concatenate((y_train, select_truths))
    elif dataset == 'fashionmnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_retrain = np.concatenate((X_train, select_images))
        y_retrain = np.concatenate((y_train, select_truths))
    elif dataset == 'cifar':
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        X_retrain = np.concatenate((X_train, select_images),axis=0)
        y_retrain = np.concatenate((y_train, select_truths),axis=0)
    elif dataset == 'svhn':
        X_train=np.load('./dataset/svhn/x_train.npy')
        y_train=np.load('./dataset/svhn/y_train.npy')
        select_truths = np_utils.to_categorical(select_truths, n_classes)
        X_retrain = np.concatenate((X_train, select_images),axis=0)
        y_retrain = np.concatenate((y_train, select_truths),axis=0)
    shuffle_index = np.random.permutation(np.arange(len(X_retrain)))
    X_retrain_shuffled = X_retrain[shuffle_index, ...]
    y_retrain_shuffled = y_retrain[shuffle_index]
        
    return X_retrain_shuffled, y_retrain_shuffled,remain_images, remain_truths

preprocess_dic = {
    'cifar': cifar_preprocessing,
    'mnist': mnist_preprocessing,
    'fashionmnist': mnist_preprocessing,
    'svhn': svhn_preprocessing
}
preprocess_dic1 = {
    'cifar': cifar_preprocessing1,
    'mnist': mnist_preprocessing1,
    'fashionmnist': mnist_preprocessing1,
    'svhn': cifar_preprocessing1
}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', help="dataset",
                            choices=['mnist', 'cifar','fashionmnist','svhn','mnist1'], default='mnist')
    parser.add_argument('-strategy', help="the retrain data selection strategy, used the metric from RobOT", choices=[ 'high','low','best','kmst'], default='best')
    parser.add_argument('-random', help="the metric used to select retrain data", choices=[ 'random','pcs','fol'], default='fol')
    parser.add_argument('-num', help="the number of retrain data added into training dataset", type=int, default='600')

    args = parser.parse_args()
        
    batch_size = 64
    n_classes = 10

    preprocess = preprocess_dic[args.dataset]
    preprocess1 = preprocess_dic1[args.dataset]

    crash_path='./crashes'
    label_path='./crash_labels'
    crash=get_list(crash_path)
    label=get_list(label_path)

    x=get_list(crash_path)
    y=get_list(label_path)
    x_numpy=get_numpy(crash_path) 
    y_numpy=get_ynumpy(label_path) 
    path=crash_path
    xi_path=crash_path  
    
    x_train,y_train,_,_=select_function(x, y,xi_path,x_numpy,y_numpy,select_ratio=0.1, dataset=args.dataset,s=args.random,ss=args.strategy,set_select_number=args.num)

    if args.dataset == 'mnist':
        model = load_model("./models/lenet5.h5")
        (_, _), (x_test, y_test) = mnist.load_data()
        x_train=mnist_preprocessing(x_train)
        x_test=mnist_preprocessing(x_test)
        with np.load("./Testdata.npz") as f:   #the composed test dataset, if it needs, do the preprocessing
            xtest, ytest = f['advs'], f['labels']

    elif args.dataset == 'fashionmnist':
        model = load_model("./models/lenet5.h5")
        (_, _), (x_test, y_test) = fashion_mnist.load_data()
        x_train=mnist_preprocessing(x_train)
        x_test=mnist_preprocessing(x_test)
        with np.load("./Testdata.npz") as f:   #the composed test dataset, if it needs, do the preprocessing
            xtest, ytest = f['advs'], f['labels']

    elif args.dataset == 'svhn':
        model = load_model("./SVHN/cnn_model.h5")
        x_test=np.load('./svhn/x_test.npy')
        y_test=np.load('./svhn/y_test.npy')
        x_train=svhn_preprocessing(x_train)
        x_test=svhn_preprocessing(x_test)
        with np.load("./Testdata.npz") as f:   #the composed test dataset, if it needs, do the preprocessing
            xtest, ytest = f['advs'], f['labels']

    elif args.dataset == 'cifar':
        model = load_model("/data/zyh/deephunter/deephunter/profile/cifar10/models/resnet.h5")
        (_, _),(x_test, y_test) = cifar10.load_data()
        x_train = cifar_preprocessing(x_train)
        x_test = cifar_preprocessing(x_test)      
        with np.load("./Testdata_c.npz") as f:   #the composed test dataset, if it needs, do the preprocessing
            xtest, ytest = f['advs'], f['labels']
        
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    score = model.evaluate(x_test, y_test)
    print("original accuracy:",score[1])

    model.fit(x_train, y_train, epochs=10, batch_size=batch_size,validation_split=0.1, verbose=1)
    score = model.evaluate(x_test, y_test)
    print("accuracy on original test dataset:",round(score[1],4))
    score1 = model.evaluate(xtest, ytest)
    print("accuracy on the composed test dataset:",round(score1[1],4))