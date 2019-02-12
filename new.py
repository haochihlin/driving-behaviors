import snakeoil3_gym 
import numpy as np 
import random as rd 
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


def block_driving(clients, target_speed, step):
	# env = TorcsEnv(vision=False, throttle=True, gear_change=False, main=1)

	velocities_range = [[100, 120], [70, 90], [40, 70]]

	for c in clients:
		c.get_servers_input(0)

	init_track_pos = [c.S.d['trackPos'] for c in clients]

	# for step in range(maxSteps,0,-1):
	for i, c in enumerate(clients):
		c.get_servers_input(step) 
		drive_example(c, target_speed, init_track_pos[i])
		c.respond_to_server()





def lane_driving(clients, target_speed, step):

	init_track_pos = 0.75

	# for step in range(maxSteps,0,-1):
	for i, c in enumerate(clients):
		c.get_servers_input(step)
		drive_example(c, target_speed, init_track_pos)
		c.respond_to_server()

	if step%100 == 0:
		init_track_pos *= -1


def main():
	maxSteps = 10000

	clients = get_n_clients(19, 3101)
	velocities_range = [[100, 110], [80, 90], [60, 70], [45, 55], [35, 44]]
	for step in range(maxSteps, 0, -1):
		for i in range(5):
			rand = np.random.choice([0, 1])
			start = i*4
			end = (i+1)*4
			if i == 4:
				end = 19
			target_speed = np.random.choice(range(velocities_range[i][0], velocities_range[i][1]))
			if rand == 1:
				block_driving(clients[start:end], target_speed, step)
			else:
				lane_driving(clients[start:end], target_speed, step)

	for c in clients:
		c.shutdown()




def drive_traffic(port, seed):
	velocities_range = [[100, 110], [80, 90], [60, 70], [45, 55], [35, 44]]
	env = TorcsEnv(vision=False, throttle=True, gear_change=False, main=1) 
	c = snakeoil3_gym.Client(p=port)
	np.random.seed(seed)
	episode_count = 1000000
	maxSteps = 1000
	episode_wise_behavior = [np.random.choice([0, 1]) for i in range(episode_count)]
	episode_wise_speed = [np.random.choice(range(velocities_range[int((port-3101)/4)][0], velocities_range[int((port-3101)/4)][1])) for i in range(episode_count)]
	lane_change = 0.75
	c.get_servers_input(0)
	init_track_pos = c.S.d['trackPos']
	for i in range(episode_count):
		for step in range(maxSteps,0,-1):
			c.get_servers_input(step)
			if c.S.d == []:
				break
			if episode_wise_behavior[i] == 0:
				drive_example(c, episode_wise_speed[i], init_track_pos)
				c.respond_to_server()
			else:
				#c.get_servers_input(step)
				drive_example(c, episode_wise_speed[i], lane_change)
				c.respond_to_server() 

				if (maxSteps - step)%50 == 0:
					lane_change *= -1
		# ob, client = env.reset(client=c, relaunch=True) 


if __name__ == "__main__":
	worker_threads = []
	ports = range(3101, 3120)
	seeds = [8, 16, 32, 64, 128]
	seeds = [seeds[int(i/4)] for i in range(19)]
	for i in range(19):
		worker_work = lambda: (drive_traffic(ports[i], seeds[i]))
		t = threading.Thread(target=(worker_work))
		print("active thread count is: " + str(threading.active_count()) + "\n")
		t.start()
		sleep(0.5)
		worker_threads.append(t)

