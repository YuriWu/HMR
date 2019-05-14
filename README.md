# HMR
This repository (currently) contains Python3 code to reproduce experiments of HMR method described in 
our ICML'19 paper **Heterogeneous Model Reuse via Optimizing Multiparty Multiclass Margin**.

## Reproduce toy example
Please check `HMR_toy_example.ipynb` in `ipynb` folder.

## Reproduce benchmark experiment on Fashion-MNIST
Please notice currently I have only tested on Python 3.6.5 on Windows 10 64-bit with the following dependencies:

    tensorflow == 1.9.0
    numpy == 1.14.3
    matplotlib ==  2.2.2
    seaborn == 0.9.0
    pandas == 0.23.0

Please download the `fashion_mnist` folder and run `run_all.bat` (for Windows) or `run_all.sh` (for Linux).

The general pipeline is:

1. `train_local_models.py` trains local models according to different settings described in config files in folder `config`, and saves the trained models into folder `model`.
2. `HMR.py` loads pre-trained local models, and runs our HMR method on different random seeds. Then saves the experimental results into folder `exp`.
3. `plot_exp_result.py` collects all the experimental results and plots the figure we used in our paper. The figure will be saved as `figure.pdf`.

A full run may take about 1 day on my single Titan Xp GPU.

## Reproduce multi-lingual handwriting experiment
We regret to say that we cannot provide everything to reproduce this experiment because some datasets are not allowed to share publicly on github. Please read the details provided in our supplementary file. Basically speaking, we use the same code as Fashion-MNIST to run the multi-lingual experiment. There are some "dirty work" to clean up and rescale all the datasets to 64*64 images, and you need to implement different `API_*.py` files to handle heterogeneous network structures. Feel free to contact me if you are interested in reimplement this experiment.

---------------

If you have questions or comments about anything related to this work, please
do not hesitate to contact [Xi-Zhu Wu](http://lamda.nju.edu.cn/wuxz/).
