#!/bin/bash

 
apt-get install -y x11-apps xauth xfonts-base
apt-get install -y libgtk-3-dev
apt-get install -y qt5-default
apt-get install -y fonts-dejavu-core



# apt-get update 
# apt install -y wget git


# bash Anaconda3-2021.05-Linux-x86_64.sh -b
# echo -e "export PATH=$PATH:/root/anaconda3/bin" >> ~/.bashrc && source ~/.bashrc
# conda init bash && source ~/.bashrc
 
# ############ GENERAL ENV SETUP ############
# echo New Environment Name:
# read envname

# echo Creating new conda environment $envname 
# conda create -n $envname python=3.8 -y -q

# eval "$(conda shell.bash hook)"
# conda activate $envname

# echo
# echo Activating $envname
# if [[ "$CONDA_DEFAULT_ENV" != "$envname" ]]
# then
#     echo Failed to activate conda environment.
   
# fi


# ############ REQUIRED DEPENDENCIES (PYBULLET) ############
# echo Installing dependencies...

# # Install other PIP Dependencies
# pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
# pip install setuptools==65.5.0
# pip install wheel==0.38.4
# pip install torchdiffeq


# echo
# echo
# echo Successfully installed.
# echo
# echo To activate your environment call:
# echo conda activate $envname
 
