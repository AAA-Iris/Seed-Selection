# Seed Selection for Testing Deep Neural Networks

# Installation
A virtual environment is strongly recommended to ensure the requirements may be installed safely.
We test DeepHunter and TensorFuzz based on Python 3.6 and Tensorflow 1.14.0. 
To install the requirements, run:
```
pip install -r requirements.txt
```
Since we integrate the method of tes case selection in RobOT, we extend the environment of this method to ensure the consistency between our results and the original result, so the environment of the retraining process is Python 3.6 and Tensorflow 2.2.0.
You could install the tensorflow-gpu environment by using conda.
```
conda create -n tf2-gpu tensorflow-gpu==2.2.0
```

# Files
- models - models used on different datasets, the suffix '_1' represents the model uses the same architecture but with different parameters.
- selected_seeds - the selected seeds under different configurations, the name of file 'strategy_number' represents the seed selection strategy and the number of selected seeds.

coverage-guided-optimization.py is to select seeds using coverage. 

uncertainty-guided-optimization.py is to select seeds using uncertainty metric.

retrain.py is to select test cases for model retraining. 

# Usage
## Example of Running the Code:

```
python coverage-guided-optimization.py
       -seednum 100
       -coverage ../coverage
       -seed ../mnist_seeds
       -out ../optimized_seeds
```

The meanings of the options are:

1. `-seednum` determines how many seeds are selected to be the initial seeds. If you set this number that equals to the number of whole original seed corpus, you can obtain the optimized seed set with the smallest size. If you want to obtain the specific number of seeds, rerun this code till it reaches the expected seed number.
2. `-coverage` is the path of coverage information, the directory preserves the coverage information of all original seeds, it contains several files which preserve coverage information of single seed with array format in each file.
3. `-seed` is the path of seed corpus, the directory contains several files which preserve single seed with array format in each file.
4. `-out` is the output directory, preserve the selected seeds.

```
python uncertainty-guided-optimization.py
       -i ../mnist_seeds
       -num 100
```

The meanings of the options are:

1. `-i` determines which model to be tested, where the 7 pre-profiled files are: `vgg16`, `resnet20`, `mobilenet`, `resnet50`, `lenet1`, `lenet4`, `lenet5`. To be started easily, we integrated 7 models, i.e., their profiling files can be found by DeepHunter. It is fairly possible to test other models; in such scenarios, please use `utils/Profile.py` to profile the new model firstly. Then, it is also easy to integrate the new model and the profiling file in DeepHunter.  
2. `-criteria` selects different testing criteria; possible values are: `kmnc`, `nbc`, `snac`, `bknc`, `tknc`, `nc`.  
3. `-random` determines whether applying random testing (1) or coverage-guided testing (0); defaults to 0.
4. `-select` chooses the selection strategies (when `-random` is 0) from `uniform`, `tensorfuzz`, `deeptest`, `prob`.  

# Coming soon
More details would be included soon. 
