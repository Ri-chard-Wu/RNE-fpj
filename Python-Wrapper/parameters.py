


import numpy as np

class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            value = self[key]
            if isinstance(value, dict):
                return AttrDict(value)
            return value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


 

train_args = AttrDict({
            'seed': 69,
            'batch_size': [32, 256][1],
            'num_steps': int(1e9),
            
            'replay_size': 250000,
           
            'update_interval': 1,

            'ckpt_dir': 'ckpt',
          
            'load_model': 'model_300.pt',
            'load_mem': 'mem.h5',

            'save_model_interval': 50,             
            'save_mem_interval': 250,
            'log_reward_interval': 1,
            'log_loss_interval': 1,

            'update_n': 32,

            'prefill_n': [1, 64, 512][0],
            'return_steps': 10, 
            'gamma': 0.99,
            'frame_skip': 1,

            'start_priority_exponent': [0.55, 0.89][1],            
            'start_importance_exponent': [0.55, 0.89][1],
            'end_priority_exponent': 0.9,
            'end_importance_exponent': 0.9,
            'prioritization_steps': 100
        })


sac_args = AttrDict({ 
    'tau': 0.005,
    'lr': 0.0003,  
    'hidden_size': 512, 
    'input_dim': 32    
})


action_low = np.array([0.0, 0.0])
action_high = np.array([0.5, 0.5])

action_dim = 2
obs_shape = (64, 64, 3)


DEVICE = "cpu"

