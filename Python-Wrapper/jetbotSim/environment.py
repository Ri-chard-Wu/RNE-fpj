import sys
sys.path.append('./jetbotSim')
import numpy as np
import cv2
import websocket
from websocket import create_connection
import threading
import time
import config
from PIL import Image
import struct
import json 
import time
from parameters import action_dim, obs_shape, DEVICE
from collections import deque

class Robot():
    def __init__(self):    
        self.ws = None
        self._connect_server(config.ip, config.actor)
        self._left_motor = 0
        self._right_motor = 0
        self.reset()

    def _connect_server(self, ip, actor):
        self.ws = create_connection("ws://%s/%s/controller/session"%(ip, actor))
        time.sleep(1)   #wait for connect
    
    def _move_to_wheel(self, value):
        length = 2 * np.pi * config.wheel_rad 
        angular_vel = 360 * (1000*value / length)
        return angular_vel

    # Control Command
    def set_left_motor(self, value):
        left_ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':left_ang, 'rightMotor':0.0, 'flag':1})
        self.ws.send(jsonStr)

    def set_right_motor(self, value):
        right_ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':0.0, 'rightMotor':right_ang, 'flag':2})
        self.ws.send(jsonStr)
    
    def set_motor(self, value_l, value_r):
        left_ang = self._move_to_wheel(value_l)
        right_ang = self._move_to_wheel(value_r)
        jsonStr = json.dumps({'leftMotor':left_ang, 'rightMotor':right_ang, 'flag':4})
        self.ws.send(jsonStr)
    
    def add_motor(self, value_l, value_r):
        left_ang = self._move_to_wheel(value_l)
        right_ang = self._move_to_wheel(value_r)
        jsonStr = json.dumps({'leftMotor':left_ang, 'rightMotor':right_ang, 'flag':3})
        self.ws.send(jsonStr)

    def forward(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':ang, 'rightMotor':ang, 'flag':4})
        self.ws.send(jsonStr)
    
    def backward(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':-ang, 'rightMotor':-ang, 'flag':4})
        self.ws.send(jsonStr)

    def left(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':-ang, 'rightMotor':ang, 'flag':4})
        self.ws.send(jsonStr)

    def right(self, value):
        ang = self._move_to_wheel(value)
        jsonStr = json.dumps({'leftMotor':ang, 'rightMotor':-ang, 'flag':4})
        self.ws.send(jsonStr)

    def stop(self):
        jsonStr = json.dumps({'leftMotor':0.0, 'rightMotor':0.0, 'flag':4})
        self.ws.send(jsonStr)
    
    def reset(self):
        jsonStr = json.dumps({'leftMotor':0.0, 'rightMotor':0.0, 'flag':0})
        self.ws.send(jsonStr)




def process_observation(screen): 
 
    img = Image.fromarray(screen)
    img = img.resize(obs_shape[:2], Image.BILINEAR)
    img = np.array(img) # (64, 64, 3) 

    return np.transpose(img, (2, 0, 1)) # (3, 64, 64)
 



class Env():
    def __init__(self):    
        self.ws = None
        self.wst = None
        self._connect_server(config.ip, config.actor)
        self.buffer = None
        self.on_change = False

        self.robot = Robot()
        
        self.k = 4
        self.recent_rewards = deque(maxlen=self.k) 

        self.recent_bads = deque(maxlen=8) 

        self.i = 0
        
        for _ in range(self.k):
            self.recent_rewards.append(1)

 
    def _connect_server(self, ip, actor):
        self.ws = websocket.WebSocketApp("ws://%s/%s/camera/subscribe"%(ip, actor), 
                                         on_message = lambda ws, msg: self._on_message_env(ws, msg))
        
        self.wst = threading.Thread(target=self.ws.run_forever)
        self.wst.daemon = True
        self.wst.start()
        time.sleep(1)   #wait for connect
    
    def _on_message_env(self, ws, msg):

        self.buffer = msg
        self.on_change = True        


    def wait_return(self): 
        while True: 
            if self.buffer is not None and self.on_change:

                nparr = np.fromstring(self.buffer[5:], np.uint8)
                img_ = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                img = process_observation(img_.copy())

                l = 12
                # print(f'\rr: {np.mean(img[:, -3:, l:-l]).round(3)}')
                # if(np.mean(img[2, -3:, :]) > 50):

                is_bad1 = np.mean(img[:, -2:, l:-l]) > 29

                r = np.mean(img[2, -3:, :]).round(3)
                rgb = np.mean(img[:, -3:, :]).round(3)
                
                is_collide = r > 45 and r > rgb

                # print(f'[{self.i}] all: {np.mean(img[:, -3:, l:-l]).round(3)}, r: {np.mean(img[2, -3:, l:-l]).round(3)}')


                # if(is_bad):
                #     print(f'[{self.i}] collision!')
                #     self.i+=1
                self.recent_bads.append(int(is_bad1))
                # print(f'\rrecent_bads: {self.recent_bads}')

                reward = int.from_bytes(self.buffer[:4], 'little')

                self.on_change = False

                self.recent_rewards.append(reward)
              
                is_bad2 = not np.any(self.recent_rewards)
                if(is_bad2): reward = -1

                # done = np.all(self.recent_bads) # and is_bad2
                # done = False
                done = is_collide

                # if(done):
                    # print(f'[{self.i}] done!')
                    # self.i+=1
                    # time.sleep(3)
                    # cv2.imwrite(f'output/{self.i}.png', np.transpose(img, (1,2,0)))

                return img, reward, done, None
       

    def reset(self):

        self.robot.reset()
        
        for _ in range(self.k):
            self.recent_rewards.append(1)

        return self.wait_return()[0]


    def step(self, a):
        
        self.robot.set_motor(a[0], a[1])

        return self.wait_return()