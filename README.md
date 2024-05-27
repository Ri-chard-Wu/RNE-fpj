

# SAC for jetbot path tracking + obstacle avoidance

<p align="center">
  <img src="Docs/gif/demo.gif">
</p> 

This repository contains code to set up a reinforcement learning agent using the Soft Actor-Critic algorithm (SAC) [https://arxiv.org/pdf/1801.01290.pdf] for jetbot path tracking + obstacle avoidance.


## Train model  
First navigate to the `Python-Wrapper` directory, the run the following command:
```bash
python train.py
```

## Pre-trained weights  

Pre-trained models are contained in the `Python-Wrapper/ckpt` directory. To use a pre-trained weights, first rename the name of the ckeckpoint file to have a ".pt" suffix, then in `Python-Wrapper/parameters.py` set `load_model` to the name of the checkpoint file.