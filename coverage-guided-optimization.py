from pulp import *
import os
import numpy as np
from keras.models import load_model
import random
import argparse

def Select_seeds(seedMap):   
    '''
    input: seedMap: a numpy of (n*m) {0/1}
    output: resSeed: a list of selected seeds
            neuronList: all neurons to be covered
    '''
    
    Variable=[]
    #built variable for each x_i
    for i in range(seedMap.shape[0]):
        Variable.append(LpVariable("Seed+"+str(i),lowBound=0,upBound=1,cat=LpBinary))
    
    neuronList=[]
    #built constraints for each neuron
    LpBase=LpProblem(name='select_seeds',sense=LpMinimize)
    
    for j in range(seedMap.shape[1]):
        flag=np.sum(seedMap[:,j])
        if flag!=0:
            #print(j)
            neuronList.append(j)
            x=[]
            y=[]
            for i in range(seedMap.shape[0]):
                if seedMap[i][j]==1:
                    x.append(Variable[i])
                    y.append(1)
            c=LpConstraint(LpAffineExpression([(x[i],y[i]) for i in range(len(x))]), LpConstraintGE, str(j),1)
            LpBase+=c
    
    #add the optimized target
    LpBase+=lpSum([Variable[i] for i in range(seedMap.shape[0])])
    #try to solve
    try:
        solver=getSolver('PULP_CBC_CMD',msg=0,timeLimit=6000)
        resl=LpBase.solve(solver)
        status = LpStatus[LpBase.status]
    except:
        print("Wrong")
        return None

    #get the answer
    resSeed=[]
    if status=='Optimal':
        for v in LpBase.variables():
            if v.varValue==1:
                st=v.name
                st=st.replace("_"," ")
                st=st.split()
                st=st[1]
                resSeed.append(int(st))
    else:
        print(str(status))
    
    return resSeed, neuronList

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coverage guided seed selection')
    parser.add_argument('-seednum', help="the expected number of seeds", type=int, default=10000)
    args = parser.parse_args()  
    file_dir='./coverage'   #path of coverage information
    file='./seed-corpus'    #path of seed corpus
    tasks=os.listdir(file_dir)
    tasks.sort(key= lambda x: int(x[:-4])) 
    if not os.path.exists('./optimized_seeds'):
        os.makedirs('./optimized_seeds')
    stage=[[]]
    i=0
    for c_n in tasks:
        if not os.path.isdir(c_n): 
            f = np.load(file_dir+"/"+c_n)
            n_num=f.shape[0]
            f=f.reshape(1,n_num)
            if c_n in os.listdir('./optimized_seeds'):
                f=np.zeros((1,n_num))
            if i==0:
                stage=f
                i=i+1
            else:
                stage=np.append(stage,f,axis=0) #get the coverage targets
    print(stage.shape)

    resSeed, neuronList=Select_seeds(stage) 
  
    num=len(os.listdir('./optimized_seeds'))  
    if len(resSeed)<= (args.seednum-num):   
        for i in resSeed: 
            np.save(r'./optimized_seeds/{}'.format(tasks[i]),np.load(file+"/"+tasks[i]))   
    else:
        label=random.sample(resSeed,args.seednum-num)
        for i in label:        
            np.save(r'./optimized_seeds/{}'.format(tasks[i]),np.load(file+"/"+tasks[i]))
    print("seed:",resSeed)
    print("seed number：",len(resSeed))
    #print("neuron:",neuronList)
    print("neuron number：",len(neuronList))
    print(len(os.listdir('./optimized_seeds')))




