UPPAAL-BN - Code
-----------------------------------
This repository contains the python code for reproducing the experiments described in the paper A general framework for defining and optimizing robustness, by  Manfred Jaeger, Kim G. Larsen, and Alessandro Tibo.

Python Requirments
------------------
	bnlearn (please follow the instruction on https://github.com/erdogant/bnlearn)
  numpy
  scikit-learn
  scipy
  tqdm
  
UPPAAL
------
All the scripts assume that UPPAAL (http://www.uppaal.org) is installed in your computer and all the file `verifyta` is included in your $PATH.
	
Run the Experiments for ``Boat Crossing''
-------------------------------------
	$ cd examples/boat
	$ python train.py cifar10 (mnist, fashion_mnist, svhn)
