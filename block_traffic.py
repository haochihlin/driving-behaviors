import snakeoil3_gym 
import numpy as np 
import random as rd 
# from gym_torcs import TorcsEnv
import os

from gym_torcs import TorcsEnv
import threading
import multiprocessing
from time import *

PI= 3.14159265359

def get_n_clients(N, start_port):
	clients = []
	for i in range(N):
		c = snakeoil3_gym.Client(p=start_port+i)
		clients.append(c)
	return clients



def drive_example(c, target_speed, init_pos):
	u'''This is only an example. It will get around the track but the
	correct thing to do is write your own `drive()` function.'''
	S,R= c.S.d,c.R.d
	# target_speed=1000
	min_dist = 10
	# Steer To Corner
	R[u'steer']= S[u'angle']*10 / PI
	# Steer To the old pose
	R[u'steer'] += (init_pos - S[u'trackPos'])*.3

	if S[u'angle']*180/PI < 5:
		R[u'steer'] = np.clip(R[u'steer'], -.1, .1)

	# print(R[u'steer'])

	R[u'accel'] = 0.5

	if target_speed-10 < S[u'speedX']:
	    R[u'accel'] = 0.01

	if target_speed <= S[u'speedX']:
	    R[u'accel'] = 0.


	opponents = S['opponents']
	front = np.array([opponents[15], opponents[16], opponents[17], opponents[18], opponents[19], opponents[20]])
	back  = np.array([opponents[-3], opponents[-2], opponents[-1], opponents[0], opponents[1], opponents[2]])
	left  = np.array([opponents[6], opponents[7], opponents[8], opponents[9], opponents[10], opponents[11]])
	right = np.array([opponents[23], opponents[24], opponents[25], opponents[26], opponents[27], opponents[28], opponents[29]])


	# R[u'accel'] += np.random.normal(0, 0.1)

	# Throttle Control
	# if S[u'speedX'] < target_speed - (R[u'steer']*50):
	#     R[u'accel']+= .01
	# else:
	#     R[u'accel']-= .01
	# if S[u'speedX']<10:
	#    R[u'accel']+= 1/(S[u'speedX']+.1)


	if min(front) < 10:
		R[u'accel'] = 0.0

	if min(back) < 10:
		R[u'accel'] += 1.0
		R[u'brake'] = 0.

	# if min(front) > 100:
	# 	R[u'accel'] += 1.0

	# Traction Control System
	# if ((S[u'wheelSpinVel'][2]+S[u'wheelSpinVel'][3]) -
	#    (S[u'wheelSpinVel'][0]+S[u'wheelSpinVel'][1]) > 5):
	#    R[u'accel']-= .2

	# Automatic Transmission
	# R[u'gear']=1
	# if S[u'speedX']>50:
	#     R[u'gear']=2
	# if S[u'speedX']>80:
	#     R[u'gear']=3
	# if S[u'speedX']>110:
	#     R[u'gear']=4
	# if S[u'speedX']>140:
	#     R[u'gear']=5
	# if S[u'speedX']>170:
	#     R[u'gear']=6
	return


def block_definition(init_port, target_speed_range):
	num_episodes = 100000
	maxSteps = 10000
	episode_wise_behavior = [np.random.choice([0,1]) for i in range(num_episodes)]
	episode_wise_speed = [np.random.choice(range(target_speed_range[0], target_speed_range[1])) for i in range(num_episodes)]
	env = TorcsEnv(vision=False, throttle=True, gear_change=False, main=1) 
	lane_change = np.random.choice([0.75, -0.75])
	if init_port == 3117:
		num_clients = 3
	else:
		num_clients = 4
	clients = get_n_clients(num_clients, init_port)

	for episode in range(num_episodes):

		if episode_wise_behavior[episode] == 0:
			# Block Driving behavior 
			for c in clients:
				c.get_servers_input(0)

			init_track_pos = [c.S.d['trackPos'] for c in clients]

			for step in range(maxSteps):
				for i, c in enumerate(clients):
					try:
						c.get_servers_input(step) 
						drive_example(c, episode_wise_speed[episode], init_track_pos[i])
						c.respond_to_server()
					except:
						ob = None
						while ob is None:
							try:
								c = snakeoil3_gym.Client(p=port, vision=False)  # Open new UDP in vtorcs
								c.MAX_STEPS = np.inf

								c.get_servers_input(0)  # Get the initial input from torcs

								obs = c.S.d  # Get the current full-observation from torcs
								ob = env.make_observation(obs)

								s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
							except:
								pass	

		else:
			for step in range(maxSteps):
				for i, c in enumerate(clients):
					try:
						c.get_servers_input(step)
						drive_example(c, episode_wise_speed[episode], lane_change)
						c.respond_to_server() 
					except:
						ob = None
						while ob is None:
							try:
								c = snakeoil3_gym.Client(p=port, vision=False)  # Open new UDP in vtorcs
								c.MAX_STEPS = np.inf

								c.get_servers_input(0)  # Get the initial input from torcs

								obs = c.S.d  # Get the current full-observation from torcs
								ob = env.make_observation(obs)

								s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))
							except:
								pass

				if step%20 == 0:
					lane_change = np.random.choice([0.75, -0.75])






if __name__ == "__main__":
	worker_threads = []
	ports = [3101, 3105, 3109, 3113, 3117]
	velocities_range = [[100, 110], [80, 90], [60, 70], [45, 55], [35, 44]]

	for i in range(5):
		worker_work = lambda: (block_definition(ports[i], velocities_range[i]))
		t = threading.Thread(target=(worker_work))
		print("active thread count is: " + str(threading.active_count()) + "\n")
		t.start()
		sleep(0.5)
		worker_threads.append(t)

