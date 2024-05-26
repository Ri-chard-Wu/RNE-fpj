from jetbotSim import Env  
from PIL import Image 

import numpy as np  
import torch 

from sac import SAC
from replay_memory import PrioritizedExperienceReplay 
  
from collections import deque
import traceback
from tqdm import tqdm

from parameters import action_dim, obs_shape, DEVICE
from parameters import train_args as args
 




class EnvWrapper:
    def __init__(self, env):
        self.env = env

 

class FrameSkipWrapper(EnvWrapper):
    def __init__(self, env):
        EnvWrapper.__init__(self, env) 

    def reset(self):
        return self.env.reset()

    def step(self, action):
        step_reward = 0.0
        for _ in range(args.frame_skip): 
            obs, reward, done, info = self.env.step(action) 
            step_reward += reward
            if done: break
        return obs, step_reward, done, info



 


class Trainer():
 
    def __init__(self):
 
        env = Env() 
        self.env = FrameSkipWrapper(env)


        torch.manual_seed(args.seed)
        np.random.seed(args.seed) 
 
        self.agent = SAC()
        if 'load_model' in args: 
            self.agent.load(args.ckpt_dir, args.load_model)
        
    
        self.memory = PrioritizedExperienceReplay(args.replay_size)        
        if 'load_mem' in args: 
            self.memory.load_data(args.ckpt_dir, args.load_mem)         
        self.n_step_buffer = deque(maxlen=args.return_steps)

        self.priority_exponent =  args.start_priority_exponent
        self.importance_exponent =  args.start_importance_exponent


        self.obs = self.env.reset() # (3, 64, 64)
        # self.obs = process_observation(self.obs) # (3, 64, 64)
        
        self.vel = np.array([0.0, 0.0])
        
        self.episode_reward = 0

        self.n_episode = 0
        self.t = 0



    def calc_multistep_return(self):

        Return = 0
        t = 0
        while(1): 
            _, _, _, reward, _, mask = self.n_step_buffer[t]

            Return += args.gamma**t * reward
            if(mask < 0.5 or t==args.return_steps-1): break
            t+=1
        
        # buf format: obs, act_prev, action, reward, next_obs, mask.
       
        # obs, obs_vel, a, r, obs_next, obs_vel_next, mask, T.
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], self.n_step_buffer[0][2], Return, \
                            self.n_step_buffer[t][4], self.n_step_buffer[t][2], self.n_step_buffer[t][5], t+1
    


    def rollout_episode(self):
        
        while (1):

            action = self.agent.select_action(self.obs, self.vel) # (2,).
            assert action.shape == (action_dim,)

            next_obs, reward, done, _ = self.env.step(action)  # Step

            if(done):
                next_obs = self.env.reset()  # (3, 64, 64)
    
                if(self.n_episode % args.log_reward_interval == 0):
                    with open("rewards.txt", "a") as f: 
                        f.write(f'[{self.n_episode}] {self.episode_reward}' + '\n')
                                    
                    print(f'[{self.n_episode}] episode_reward: {self.episode_reward}')

                self.episode_reward = 0 
                self.n_episode += 1
                
                

            # next_obs = process_observation(next_obs) # (3, 64, 64)
            
            mask = float(not done)
            self.episode_reward += reward 

            data = (self.obs, self.vel, action, reward, next_obs, mask)
            self.obs = next_obs
            self.vel = action if not done else np.array([0.0, 0.0])

            
            self.n_step_buffer.append(data)

            if len(self.n_step_buffer) == args.return_steps:
                
                # obs, obs_vel, a, r, obs_next, obs_vel_next, mask, T.
                data = self.calc_multistep_return()   

                s_bt = torch.from_numpy(data[0].astype(np.float32)).to(DEVICE).unsqueeze(0)
                vel_bt = torch.from_numpy(data[1].astype(np.float32)).to(DEVICE).unsqueeze(0)
                a_bt = torch.from_numpy(data[2].astype(np.float32)).to(DEVICE).unsqueeze(0)
                r_bt = torch.from_numpy(np.array([data[3]]).astype(np.float32)).to(DEVICE).unsqueeze(0)
                s_next_bt = torch.from_numpy(data[4].astype(np.float32)).to(DEVICE).unsqueeze(0)
                vel_next_bt = torch.from_numpy(data[5].astype(np.float32)).to(DEVICE).unsqueeze(0)
                mask_bt = torch.from_numpy(np.array([data[6]]).astype(np.float32)).to(DEVICE).unsqueeze(0)
                T_bt = torch.from_numpy(np.array([data[7]]).astype(np.float32)).to(DEVICE).unsqueeze(0)
                
                assert tuple(r_bt.shape) == (1,1)
                assert tuple(mask_bt.shape) == (1,1)
                assert tuple(T_bt.shape) == (1,1)

                with torch.no_grad():
                    q1_loss, q2_loss = self.agent.td_loss(\
                            (s_bt, vel_bt, a_bt, r_bt, s_next_bt, vel_next_bt, mask_bt, T_bt))
                    priorities = self.agent.compute_priority(q1_loss, q2_loss)
                
                assert priorities.shape == (1,)

                self.memory.push(data, priorities[0], self.priority_exponent) 


            if(done): break



    def train(self):
         
                
        while(1):
            self.rollout_episode()
            print(f'\rprefile {len(self.memory)} / {args.prefill_n}', end="")
            if(len(self.memory) >= args.prefill_n): break
            

        print()


        priority_delta = (args.end_priority_exponent -\
                            args.start_priority_exponent) / args.prioritization_steps
        importance_delta = (args.end_importance_exponent -\
                            args.start_importance_exponent) / args.prioritization_steps


        for _ in range(args.num_steps):
            
            self.t += 1

            self.rollout_episode() 

            if len(self.memory) > args.batch_size: 
                
                for _ in range(args.update_n):
                    batch, idxs, importance_weights = self.memory.sample(\
                                            args.batch_size, self.importance_exponent)
                    priorities, policy_loss, ent_loss =\
                                            self.agent.update_parameters(batch, importance_weights)
                    
                    self.memory.update_priorities(idxs, priorities, self.priority_exponent)

 
                self.priority_exponent += priority_delta
                self.priority_exponent = min(args.end_priority_exponent, self.priority_exponent)

                self.importance_exponent += importance_delta
                self.importance_exponent = min(args.end_importance_exponent, self.importance_exponent)


                if(self.t % args.log_loss_interval == 0):
                    
                    log = str({'policy_loss': round(policy_loss, 3), 
                                'ent_loss': round(ent_loss, 3),
                                'mem_len': len(self.memory),
                                'priority_exponent': self.priority_exponent,
                                'importance_exponent': self.importance_exponent
                                })
                    
                    # print(f'[{self.t}]' + log)

                    with open("losses.txt", "a") as f: 
                        f.write(f'[{self.t}]' + log + '\n')



            if self.t % args.save_model_interval == 0:
                self.agent.save(args.ckpt_dir, f"model_{self.t}.pt")

            if self.t % args.save_mem_interval == 0:
                self.memory.save_data(args.ckpt_dir, f"mem.h5")


 

if __name__ == "__main__": 
    

    try:
        trainer = Trainer()
        trainer.train()
    except Exception:
        trainer.env.reset()
        print(f'>> Exept:')
        print(traceback.format_exc())