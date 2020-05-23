# Code for the JSTARS2020 paper 'Delving into Classifying Hyperspectral Images via Graphical Adversarial Learning'.
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

## *Citation

@ARTICLE{Wang20GAL,

  author={G. {Wang} and P. {Ren}},

  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 

  title={Delving into Classifying Hyperspectral Images via Graphical Adversarial Learning}, 

  year={2020},

  volume={},

  number={},

  pages={}
  }

