import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os
from model import GaussianPolicy, QNetwork
import logging
from pathlib import Path

from parameters import action_low, action_high, action_dim, DEVICE
from parameters import sac_args as args
from parameters import train_args
 
# torch.autograd.set_detect_anomaly(True)

def action_rescale(action):
    action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)
    return np.clip(action, action_low, action_high)



def soft_update(target, source, tau):
 
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
 
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


 
class SAC(object):
 
    def __init__(self):
          
        # self.critic = QNetwork(args.input_dim, action_dim, args.hidden_size).to(DEVICE)
        # self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic1 = QNetwork(args.input_dim, action_dim, args.hidden_size).to(DEVICE)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=args.lr)
 
        self.critic2 = QNetwork(args.input_dim, action_dim, args.hidden_size).to(DEVICE)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=args.lr)
 
        self.critic1_target = QNetwork(args.input_dim, action_dim, args.hidden_size).to(DEVICE)
        self.critic2_target = QNetwork(args.input_dim, action_dim, args.hidden_size).to(DEVICE)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)
 
        self.target_entropy = -torch.Tensor([action_dim]).to(DEVICE).item() # (,).
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE) # (1,).
        self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
        self.alpha = self.log_alpha.exp() # (1,).

        assert tuple(self.alpha.shape) == (1,)

        self.policy = GaussianPolicy(args.input_dim, action_dim, args.hidden_size).to(DEVICE)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
 
 
 
    def select_action(self, obs, vel, eval=False):
        '''
            state: (64, 64, 3).
        '''

        obs = torch.FloatTensor(obs).to(DEVICE).unsqueeze(0) # (1, 3, 64, 64).
        vel = torch.FloatTensor(vel).to(DEVICE).unsqueeze(0) # (1, 2).

        assert tuple(obs.shape) == (1, 3, 64, 64)

        if eval == False:
            action, _, _ = self.policy.sample(obs, vel) # sampled.
        else:
            _, _, action = self.policy.sample(obs, vel) # mean.

        action = action.detach().cpu().numpy() # (1, 2).
        assert  action.shape == (1, action_dim)


        return action_rescale(action[0]) # (2,).



    def td_loss(self, data):

        # s_bt, a_prev_bt, a_bt, r_bt, s_next_bt, mask_bt, T_bt = data
        s_bt, vel_bt, a_bt, r_bt, s_next_bt, vel_next_bt, mask_bt, T_bt = data

        # b = tuple(s_bt.shape)[0]

        with torch.no_grad():
            next_a_bt, next_log_pi_bt, _ = self.policy.sample(s_next_bt, vel_next_bt)
            
            # assert tuple(next_log_pi_bt.shape) == (b, 1)
            # assert tuple(next_a_bt.shape) == (b, action_dim)

            qf1_next_target = self.critic1_target(s_next_bt, next_a_bt) # (b, 1), (b, 1). 
            qf2_next_target = self.critic2_target(s_next_bt, next_a_bt) # (b, 1), (b, 1). 
               
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi_bt # (b, 1).

            # assert tuple(min_qf_next_target.shape) == (b, 1) 

            next_q_value = r_bt + mask_bt * train_args.gamma**T_bt * (min_qf_next_target) # (b, 1).

            # assert tuple(next_q_value.shape) == (b, 1)
       
        qf1 = self.critic1(s_bt, a_bt) # (b, 1), (b, 1).        
        qf2 = self.critic2(s_bt, a_bt) # (b, 1), (b, 1).        

        q1_loss = (qf1-next_q_value).pow(2) # (b, 1).
        q2_loss = (qf2-next_q_value).pow(2) # (b, 1).

        return q1_loss.squeeze(1), q2_loss.squeeze(1)

  
 

    def compute_priority(self, q1_loss, q2_loss):  
        
        q_loss = torch.sqrt(2.0 * torch.max(q1_loss, q2_loss)) # (b,).
        return (q_loss.detach() + 1e-6).cpu().numpy() # (b,).
        

           


    def update_parameters(self, batch, importance_weights):
 
        s_bt, vel_bt, a_bt, r_bt, s_next_bt, vel_next_bt, mask_bt, T_bt = batch

         
        s_bt = torch.from_numpy(s_bt.astype(np.float32)).to(DEVICE)
        vel_bt = torch.from_numpy(vel_bt).to(DEVICE) 
        a_bt = torch.from_numpy(a_bt).to(DEVICE)
        r_bt = torch.from_numpy(r_bt).to(DEVICE).unsqueeze(1) 
        s_next_bt = torch.from_numpy(s_next_bt.astype(np.float32)).to(DEVICE)
        vel_next_bt = torch.from_numpy(vel_next_bt).to(DEVICE) 
        mask_bt = torch.from_numpy(mask_bt).to(DEVICE).unsqueeze(1)
        T_bt = torch.from_numpy(T_bt.astype(np.float32)).to(DEVICE).unsqueeze(1) 

        importance_weights = torch.from_numpy(importance_weights).to(DEVICE)

        # b = tuple(s_bt.shape)[0]

        # q1_loss, q2_loss = self.td_loss((s_bt, a_bt, r_bt, s_next_bt, mask_bt, T_bt)) # (b,), (b,).
 
 
        with torch.no_grad():
            next_a_bt, next_log_pi_bt, _ = self.policy.sample(s_next_bt, vel_next_bt)
             
            qf1_next_target = self.critic1_target(s_next_bt, next_a_bt) # (b, 1), (b, 1). 
            qf2_next_target = self.critic2_target(s_next_bt, next_a_bt) # (b, 1), (b, 1). 


            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_pi_bt # (b, 1).
 
            next_q_value = r_bt + mask_bt * train_args.gamma**T_bt * (min_qf_next_target) # (b, 1).
 
        qf1 = self.critic1(s_bt, a_bt) # (b, 1), (b, 1).        
        qf2 = self.critic2(s_bt, a_bt) # (b, 1), (b, 1).        

        q1_loss = (qf1-next_q_value).pow(2) # (b, 1).
        q2_loss = (qf2-next_q_value).pow(2) # (b, 1).


        q1_loss, q2_loss = q1_loss.squeeze(1), q2_loss.squeeze(1)
    


        self.critic1_optim.zero_grad()
        torch.mean(q1_loss * importance_weights).backward() 
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        torch.mean(q2_loss * importance_weights).backward() 
        self.critic2_optim.step()
        

        a, log_pi, _ = self.policy.sample(s_bt, vel_bt) # (b, act_dim), (b, 1).

        # assert tuple(log_pi.shape) == (b, 1)
        # assert tuple(a.shape) == (b, action_dim)


        # qf1_pi, qf2_pi = self.critic(s_bt, a) # (b, 1), (b, 1).
        qf1_pi = self.critic1(s_bt, a) # (b, 1), (b, 1).        
        qf2_pi = self.critic2(s_bt, a) # (b, 1), (b, 1).        
        min_qf_pi = torch.min(qf1_pi, qf2_pi) # (b, 1).

        # assert tuple(min_qf_pi.shape) == (b, 1)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).squeeze(1) # (b,).
        # assert tuple(policy_loss.shape) == (b,)
        # policy_loss = ((log_pi) - min_qf_pi).mean() # (b,).
         
        
        self.policy_optim.zero_grad()
        torch.mean(policy_loss * importance_weights).backward()
        self.policy_optim.step()
 

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        soft_update(self.critic1_target, self.critic1, args.tau)
        soft_update(self.critic2_target, self.critic2, args.tau)
 
        return self.compute_priority(q1_loss, q2_loss), policy_loss.mean().item(), alpha_loss.item()




    def save(self, dir_name, name):

        path = os.path.join(dir_name, name)

        torch.save({
            'policy': self.policy.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha.item()
        }, path) 

        print(f'saved ckpt: {path}')   
 


    def load(self, dir_name, name):

        path = os.path.join(dir_name, name)

        checkpoint = torch.load(path)
      
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])

        self.log_alpha.data = torch.tensor(
            [checkpoint['log_alpha']],
            dtype=torch.float32,
            device=DEVICE
        )

        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

        print(f'loaded ckpt: {path}')


 
