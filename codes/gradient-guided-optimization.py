from tensorflow import keras
import random
import tensorflow as tf
import numpy as np
from keras.models import load_model
import time
from tensorflow.keras.datasets import cifar10,mnist,fashion_mnist
import argparse
from keras import backend as K
import os
def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(x_test.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(x_test.shape[0],32, 32, 3)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

def svhn_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(x_test.shape[0], 32, 32, 3)
    temp = temp.astype('float32')
    temp /= 255
    return temp

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='uncertainty guided seed selection')
    
    parser.add_argument('-seednum', help="expected number of seeds",default='100')
    parser.add_argument('-model', help='model path')
    parser.add_argument('-out', help='output directory')
    args = parser.parse_args()
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  #change if use other dataset
    
    preprocess = mnist_preprocessing   #change if use other dataset
    
    y_test = keras.utils.to_categorical(y_test, 10)

    model = keras.models.load_model(args.model)

    seeds_filter = []
    gen_img = tf.Variable(preprocess(x_test))
    with tf.GradientTape() as g:
        loss = keras.losses.categorical_crossentropy(y_test, model(gen_img))
        grads = g.gradient(loss, gen_img)

    fols = np.linalg.norm((K.eval(grads)+1e-20).reshape(x_test.shape[0], -1), ord=2, axis=1)
    seeds_rank=np.argsort(-fols)
    num=args.seednum
    seeds_filter=seeds_rank[:num]
    if not os.path.exists(args.out):
        os.makedirs(args.out)
                    
    for idx in seeds_filter:
        tmp_img = x_test[idx]
        fn="%s/%s" % (args.out, str(idx))
        np.save(fn,tmp_img)
    print(len(os.listdir(args.out)))
