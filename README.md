# Seed Selection for Testing Deep Neural Networks

# Installation
A virtual environment is strongly recommended to ensure the requirements may be installed safely.
We test DeepHunter and TensorFuzz based on Python 3.6 and Tensorflow 1.14.0. 
To install the requirements, run:
```
pip install -r requirements.txt
```
Since we integrate the method of tes case selection in RobOT, we extend the environment of this method to ensure the consistency between our results and the original result, so the environment of the retraining process is Python 3.6 and Tensorflow 2.2.0.
You could install the tensorflow-gpu environment by using conda
```
conda create -n tf2-gpu tensorflow-gpu==2.2.0
```

# Files
- models - models used on different datasets, the suffix '_1' represents the model uses the same architecture but with different parameters.
- selected_seeds - the selected seeds under different configurations, the name of file 'strategy_number' represents the seed selection strategy and the number of selected seeds.

coverage-guided-optimization.py is to select seeds using coverage. 

uncertainty-guided-optimization.py is to select seeds using uncertainty metric.

retrain.py is to select test cases for model retraining. 

# Coming soon
More details would be included soon. 
