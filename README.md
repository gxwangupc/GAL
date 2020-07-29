# Code for the JSTARS2020 article 'Delving into Classifying Hyperspectral Images via Graphical Adversarial Learning'.
---------------------------------------------
* Here I provide Tensorflow implementations for GAL, BGAC, and GAC.

* I was inspired to an extent by the two nice repositories <br> <https://github.com/KonstantinosF/Classification-of-Hyperspectral-Image> <br>
and <https://github.com/zhenxuan00/graphical-gan>. <br>
I feel grateful to the authors for providing them.

## * Environment & Main Dependencies
>CentOS Linux release 7.2.1511 (Core)<br>
>Tesla K80 Graphic Processing Units<br>
>python 2.7.15<br>
>TensorFlow 1.14.0

## * Usage
* Download hyperspectral data and add them to `./dataset`.<br>
* All the hyperparameters are in `./tflib/config.py`.<br>
Set them to what you want when running a code.<br>
* Run<br>
`python GAL.py -GPU 0` <br> 
to see GAL in the local mode on the Salinas dataset.

## * Citation

>@article{Wang20GAL, <br> 
    author={Wang, Guangxing and Ren, Peng}, <br> 
    journal={IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.}, <br> 
    title={Delving into Classifying Hyperspectral Images via Graphical Adversarial Learning}, <br> 
    year={2020}, <br> 
    volume={13}, <br> 
    number={}, <br> 
    pages={2019-2031} <br> 
  }

