# FiberTractSegmentation
ACCURATE CORRESPONDING FIBER TRACT SEGMENTATION VIA FIBERGEOMAP LEARNER

Dependencies & Installation

    python3.5
    matlab_R2019a

First, you have to make sure that matlab supports python calls:

    source activate your_python_environment
    cd R2017a/extern/engines/python
    python setup.py install
Then, run “run.py” for fiber segmentation:

    python run.py --datapath=yourpath
The default path is data/demo.vtk. 
Your data must be in Ascii vtk format.
You can find the segmentation results in the 'data/result'.

Video display of segmentation result：


![gif](https://github.com/Garand0o0/FiberTractSegmentation/tree/master/data/images/2.gif)
