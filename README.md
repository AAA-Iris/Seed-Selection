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

gradient-guided-optimization.py is to select seeds using gradient metric.

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
       -seed ../mnist_seeds
       -seednum 100
       -model ../lenet5.h5
       -out ../optimized_seeds
```

The meanings of the options are:

1. `-seed` is the input seed directory, preserve the seed corpus.
2. `-seednum` presents the expected number of selected seeds.  
3. `-model` is the path of tested model.
4. `-out` is the output directory, preserve the selected seeds.

```
conda activate tf2-gpu
python gradient-guided-optimization.py
       -seednum 100
       -model ../lenet5.h5
       -out ../optimized_seeds
```

The meanings of the options are:

1. `-seednum` presents the expected number of selected seeds.  
2. `-model` is the path of tested model.
3. `-out` is the output directory, preserve the selected seeds.

```
conda activate tf2-gpu
python retrain.py
       -dataset mnist
       -strategy best
       -random fol
       -num 600
```

The meanings of the options are:

1. `-dataset` determines the dataset that we are conducting testing on, possible choices are: `mnist`, `fashionmnist`, `svhn`, `cifar`.
2. `-strategy` determines which retrain data selection strategy is used, possible choices are: `high`, `low`, `best`, `kmst`. We use 'best' in the paper.
3. `-random` determines which metric is used to select retrain data, possible choices are: `random`, `pcs`, `fol`. We use `fol` in the paper.
4. `-num` is the number of retrain data added into training dataset.

Note that, in the code, `crash_path` represents the path of test errors, `label_path` represents the path of test error labels, the path of model and the composed test dataset need to be supplemented.

If you want to use MOO-based seed selection, just make sure the value of k1, k2 and k3, run the above code seperately.

# Coming soon
More details would be included soon. 
