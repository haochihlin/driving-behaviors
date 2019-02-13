import tensorflow as tf

import threading
import multiprocessing
from drive import *
from ddpg import *



worker_work = lambda: (block_driving(10000))
t = threading.Thread(target=(worker_work))
t.start()
sleep(0.5)


playGame(1)
