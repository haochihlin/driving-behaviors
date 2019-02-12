import threading
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import playGame_DDPG
#matplotlib inline
import os
from random import choice
from time import sleep
from time import time
import snakeoil3_gym as snakeoil3
import pymp
import sys
from drive import *
sys.path.append('./sample_DDPG_agent/')
from ddpg import *
#class Worker(object):
#    def __init__(self, name, port):

with tf.device("/cpu:0"): 
        num_workers = 1 #multiprocessing.cpu_count()
        print("numb of workers is" + str(num_workers))
        #workers = []
        #for i in range(num_workers):
        #        client = snakeoil3.Client(p=3101+i, vision=False)  # Open new UDP in vtorcs
        #        client.MAX_STEPS = np.inf
        #        workers.append("")#playGame_DDPG.playGame(f_diagnostics=""+str(i), train_indicator=1, port=3101+i))

        #with tf.Session() as sess:
        worker_threads = []
        action_dim = 3  #Steering/Acceleration/Brake
        state_dim = 65  #of sensors input
        env_name = 'Torcs_Env'
        save_location = "./weights/curriculum/reward8/" #+str(port)+"/"
        agent = DDPG(env_name, state_dim, action_dim, save_location)
        # #with pymp.Parallel(4) as p:
        # # ports =  list(range(3171, 3180))
        worker_work = lambda: (block_driving(10000))
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
        for i in range(num_workers):
          worker_work = lambda: (playGame_DDPG.playGame(f_diagnostics=""+str(i), train_indicator=0, agent=agent, port=3101+i, file_name="test.txt"))
          print("hi i am here \n")
          t = threading.Thread(target=(worker_work))
          # print("active thread count is: " + str(threading.active_count()) + "\n")
          t.start()
          sleep(0.5)
          worker_threads.append(t)
