from pulp import *
import os
import numpy as np
import argparse
import random
import shutil
import copy

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
    return resSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coverage guided seed selection')
    parser.add_argument('-seednum', help="the expected number of seeds", type=int, default=10000)
    parser.add_argument('-coverage', help='the path of coverage information')
    parser.add_argument('-seed', help='the path of seed corpus')
    parser.add_argument('-out', help='output directory')
    args = parser.parse_args()  
    file_dir=args.coverage   #path of coverage information
    file=args.seed    #path of seed corpus
    tasks=os.listdir(file_dir)
    tasks.sort(key= lambda x: int(x[:-4])) 
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    stage=[[]]
    i=0
    for c_n in tasks:
        if not os.path.isdir(c_n): 
            f = np.load(file_dir+"/"+c_n)
            n_num=f.shape[0]
            f=f.reshape(1,n_num)
            if c_n in os.listdir(args.out):
                f=np.zeros((1,n_num))
            if i==0:
                stage=f
                i=i+1
            else:
                stage=np.append(stage,f,axis=0) #get the coverage targets
    print(stage.shape)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    else:
        shutil.rmtree(args.out)
        os.makedirs(args.out)
    num=len(os.listdir(args.out))
    while num<args.seednum:
        resSeed = Select_seeds(stage)
        if len(resSeed)<= (args.seednum-num):   
            for i in resSeed: 
                fn="%s/%s" % (args.out, tasks[i])
                np.save(fn,np.load(file+"/"+tasks[i]))   
                stage[i]=0
        else:
            label=random.sample(resSeed,args.seednum-num)
            for i in label:     
                fn="%s/%s" % (args.out, tasks[i])   
                np.save(fn,np.load(file+"/"+tasks[i]))    
        num=len(os.listdir(args.out))

    print('seed number:',len(os.listdir(args.out)))
