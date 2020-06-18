UPPAAL-BN - Code
----------------
This repository contains the python code for reproducing the experiments
described in the paper From Statistical Model Checking to Run-Time Monitoring using a Bayesian Network Approach, by  Manfred Jaeger, Kim G. Larsen, and Alessandro Tibo.

Python Requirments
------------------
* bnlearn (please follow the instruction on [https://github.com/erdogant/bnlearn]())
* matplotlib
* natsort
* numpy
* pandas
* scikit-learn
* scipy
* seaborn
* tqdm
  
UPPAAL
------
All the scripts assume that UPPAAL ([http://www.uppaal.org]()) is installed
in your computer. The full path of `verifyta` of UPPAAL must be included in the config.json file.
Run the Experiments for Boat Crossing
-------------------------------------
	$ cd boat
	$ python run.py
