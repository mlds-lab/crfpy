# CRFpy

CRFpy implements maximum likelihood estimation and max margin 
learning for conditional random field using the Scikit-learn 
interface. The core of the code is the CRF class which implements 
the learning algorithms. The user is expected to implement inference 
in the desired model. An example of implementing inference for a 
specific model is in linear\_chain\_crf.py.

## Requirements

CRFpy requires a number of Python modules. These modules are listed in 
requirements.txt and can be installed using pip by running,

```
pip install -r requirements.txt
```

## Setup for linear\_chain\_crf

Inference for the linear chain crf is implemented in cython and needs to be
compiled. To compile the code, run

```
python setup.py build_ext --inplace 
```

