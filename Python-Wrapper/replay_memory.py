import random
import numpy as np
import pickle
from pickle import UnpicklingError
from pathlib import Path





import numpy as np

import h5py
import os

from collections import deque
from parameters import action_dim, obs_shape


class Tree:

    def __init__(
            self, 
            capacity # 0.25e6.       
    ):
        
        self._capacity = capacity
        self._index = 0
        self._full = False
 
        # _sum_tree: idx -> priority.
        self._sum_tree = np.zeros((2 * capacity - 1,), dtype=np.float32)
  
        self.obs      = np.zeros((capacity, obs_shape[2], *obs_shape[:2]), dtype=np.uint8)
        self.vel      = np.zeros((capacity, action_dim), dtype=np.float32)
        self.act      = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew      = np.zeros((capacity), dtype=np.float32)
        self.obs_next = np.zeros((capacity,  obs_shape[2], *obs_shape[:2]), dtype=np.uint8)
        self.vel_next      = np.zeros((capacity, action_dim), dtype=np.float32)
        self.mask      = np.zeros((capacity), dtype=np.float32)
        self.T         = np.zeros((capacity), dtype=np.uint8)
   

    def _propagate(self, index):

        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self._sum_tree[parent] = self._sum_tree[left] + self._sum_tree[right]

        if parent != 0:
            self._propagate(parent)


    def update(self, index, priority):
        self._sum_tree[index] = priority
        self._propagate(index)


    def _append(self, data):

        # obs, act, rew, obs_next, mask, T = data
        obs, vel, act, rew, obs_next, vel_next, mask, T = data

        self.obs[self._index] = obs
        self.vel[self._index] = vel
        self.act[self._index] = act
        self.rew[self._index] = rew
        self.obs_next[self._index] = obs_next
        self.vel_next[self._index] = vel_next
        self.mask[self._index] = mask
        self.T[self._index] = T


    def append(self, data, priority):
        '''
        priority: (,).
        '''
        self._append(data)
    
        self.update(self._index + self._capacity - 1, priority)
        self._index = (self._index + 1) % self._capacity
        self._full = self._full or self._index == 0


    def _retrieve(self, index, value):
        '''
        find a leaf node whose cumsum is closest to `value`, starting from `index`?
        '''

        left, right = 2 * index + 1, 2 * index + 2

        if left >= len(self._sum_tree):
            return index
        
        elif value <= self._sum_tree[left]:
            return self._retrieve(left, value)
        
        else:
            return self._retrieve(right, value - self._sum_tree[left])


    def _get_data_by_idx(self, idx):
      
        data = (
            self.obs[idx],
            self.vel[idx],
            self.act[idx],
            self.rew[idx],
            self.obs_next[idx],
            self.vel_next[idx],
            self.mask[idx],
            self.T[idx]        
        )        
        return data


    def find(self, value):

        # find a leaf node whose cumsum is closest to `value`, starting from 0?
        index = self._retrieve(0, value)

        data_index = index - self._capacity + 1

        # data, current priority, data_index, tree index.
        result = (
            # self.tree[data_index % self._capacity],
            self._get_data_by_idx(data_index % self._capacity),
            self._sum_tree[index], # current priority.
            data_index, 
            index
        )
        return result

 
    def total(self): # sum of priorities of all data.
        return self._sum_tree[0]
 

    def __len__(self):

        if(self._full):
            return self._capacity
        else:
            return self._index
    


class PrioritizedExperienceReplay:

    def __init__(self, capacity=250000):

        self.capacity = capacity
        self.tree = Tree(capacity)

    def __len__(self):      
        return len(self.tree)


    def push(self, data, priority_loss, alpha=0.55):        
 
        self.tree.append(data, priority_loss ** alpha)



    def _get_sample_from_segment(self, interval_size, i):
     

        valid = False

        while not valid:

            sample = np.random.uniform(i * interval_size, (i + 1) * interval_size) # float.

            # data, current priority, data_index, tree index.
            data, priority, idx, tree_idx = self.tree.find(sample)

            if priority != 0:
                valid = True

        return data, priority, idx, tree_idx



    def sample(self, batch_size, beta=0.55):

   
        p_total = self.tree.total() # sum of priorities of all data.
        interval_size = p_total / batch_size


        zip_data = [self._get_sample_from_segment(interval_size, i) for i in range(batch_size)]

        data, priority, ids, tree_ids = zip(*zip_data)
 
        data = map(np.array, zip(*data))
 
        probs = np.array(priority) / p_total

        importance_weights = (self.capacity * probs) ** beta
        importance_weights /= importance_weights.max()
 
 
        return data, tree_ids, importance_weights




    def update_priorities(self, ids, priority_loss, alpha=0.55):
  
        priorities = np.power(priority_loss, alpha)
        [self.tree.update(idx, priority) for idx, priority in zip(ids, priorities)]



    def save_data(self, dir_name, name):

        print('saving exp_replay...')
 
        path = os.path.join(dir_name, name)


        f = h5py.File(path, mode='w')
        data_group = f.create_group('experience_replay')
        # data_group.create_dataset('capacity', data=self.experience_replay.capacity)

        for k, v in self.tree.__dict__.items():
            
            if hasattr(v, '__len__'):
                data_group.create_dataset(k, data=v, compression="lzf")  # can compress only array-like structures
            else:
                data_group.create_dataset(k, data=v)  # can't compress scalars

        f.close()        



    def load_data(self, dir_name, name):

        print('loading exp replay...')

        path = os.path.join(dir_name, name)

        f = h5py.File(path, mode='r')
        data_group = f['experience_replay']

        for key in self.tree.__dict__:
            loaded_value = data_group[key][()]
            self.tree.__dict__.update({key: loaded_value})

        f.close()