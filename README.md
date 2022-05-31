# Pole Projet PINN

Modelisation of seismic wave propagation with Physics Informed Neural Networks. 
This project aims to simplify the usage of this new technology by proposing an intuitive and easy way to train your own PINN.

```Yes it is not a dream, you can train your own Neural Network even if you are not an ML scientist !```  

Not this one : 

![PIN](/images/pin.jpg)

But this one:

![PINN](/images/pinn.png)

# Projet structure
- models : 

    -> kernel_upgrade : implementation of NTK method

    -> model : neural network model and train step

- physics : 

    -> equation : PDE and boundary conditions

    -> initial : setting of all training points

- results

- utils : 

    -> plot : plotting functions in 1D, 2D or 3D

- root : 

    -> interface : training with Tkinter interface for parameters

    -> pinn_code : training with parser and command line for long experiments

    -> config : file for config parameters
# Installation
Simply clone the project :
```bash
git clone https://github.com/Rubiksman78/Pole_projet_PINN.git
cd school-idol-training
```
Install the requirements :
```bash
pip install -r requirements.txt
```
Install FFMPEG from their [website](https://www.ffmpeg.org/download.html)

# How to use it

## I don't like command line 
You don't like to use your command line with dozens of arguments, just use the interface we've made for you.
```
python interface.py
```
Enter the conditions and the parameters you want for your equation as well as the hyperparameters for your Neural Network.
Wait until your PINNis finished training.
Admire the result.

## You want to do more experiments
Just go to the config.py file and modify the parameters for the training or launch the main file with
```
python pinn_code.py -epochs=100 ...
```

# Coming (soon) 
Many improvements will be made in the future :
- [ ] Improving the architecture of the model used
- [ ] Giving more support for 2D and 3D modelisation
- [ ] Support for heterogenous domains and waves
