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
You can find the segmentation results in the 'data/result', including 104 vtk (103 tracts and one others.vtk)

Video display of segmentation results：

![3 00_00_00-00_00_30.gif](https://i.loli.net/2021/10/15/r6MzbBs7ito8UlC.gif)
