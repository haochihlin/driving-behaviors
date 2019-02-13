from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
import snakeoil3_gym as snakeoil3

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.999
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 65  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 4000
    max_steps = 300
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 1

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True,gear_change=False, main=1)
    ob=None
    while ob is None:
        try:
            client = snakeoil3.Client(p=3101, vision=False)  # Open new UDP in vtorcs
            client.MAX_STEPS = np.inf

            client.get_servers_input(0)  # Get the initial input from torcs

            obs = client.S.d  # Get the current full-observation from torcs
            ob = env.make_observation(obs)

            s_t = np.vstack((np.zeros([9,65]), np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))))
        except:
            pass
    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel_lanekeeping.h5")
        critic.model.load_weights("criticmodel_lanekeeping.h5")
        actor.target_model.load_weights("actormodel_lanekeeping.h5")
        critic.target_model.load_weights("criticmodel_lanekeeping.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        for j in range(max_steps):
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0], s_t.shape[1]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.4 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.0, 0.05)

            #The following code do the stochastic brake
            if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
                noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)
                noise_t[0][1] = -a_t_original[0][1]
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            # try:
            ob, r_t, done, info = env.step(j, client, a_t[0], 1)

            s_t1 = np.vstack((s_t[1:,:],np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm, ob.opponents))))

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
            # except Exception as e:
            #     print("Exception caught at port " + str(i) + str(e) )
            #     ob = None
            #     while ob is None:
            #         try:
            #             client = snakeoil3.Client(p=3101, vision=False)  # Open new UDP in vtorcs
            #             client.MAX_STEPS = np.inf
            #             client.get_servers_input(0)  # Get the initial input from torcs
            #             obs = client.S.d  # Get the current full-observation from torcs
            #             ob = env.make_observation(obs)
            #         except:
            #                 pass
            #         continue
            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss, "Velocity", ob.speedX*300)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel_lanekeeping.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel_lanekeeping.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        try:
        	if 'termination_cause' in info.keys() and info['termination_cause']=='hardReset':
        		print('Hard reset by some agent')
        		ob, client = env.reset(client=client, relaunch=True)
        	else:
        		ob, client = env.reset(client=client, relaunch=True)
			# os.system("xte 'keydown Shift_R' 'key equal' 'keyup Shift_R'")
			# os.system("xte 'keydown Shift_R' 'key equal' 'keyup Shift_R'")
        except Exception as e:
        	print("Exception caught at point B at port " + str(i) + str(e) )
        	ob = None
        	while ob is None:
                    try:
                    	client = snakeoil3.Client(p=3101, vision=False)  # Open new UDP in vtorcs
                    	client.MAX_STEPS = np.inf
                    	client.get_servers_input(0)  # Get the initial input from torcs
                    	obs = client.S.d  # Get the current full-observation from torcs
                    	ob = env.make_observation(obs)
                    except:
                    	print("Exception caught at at point C at port " + str(i) + str(e) )



    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
