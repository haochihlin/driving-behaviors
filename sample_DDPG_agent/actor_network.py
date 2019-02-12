import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 300
LAYER2_SIZE = 400
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 32
class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # create actor network
        self.a,self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

        # select trainable weights
        self.trainable_weights = self.net

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)

        # define training rules
        self.create_training_method()

        # self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())

        self.update_target()
        #self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,self.trainable_weights,-self.q_gradient_input)
        '''for i, grad in enumerate(self.parameters_gradients):
            if grad is not None:
                self.parameters_gradients[i] = tf.clip_by_value(grad, -2.0,2.0)'''
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.trainable_weights))
    def create_network(self,state_dim,action_dim):

        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float",[None,state_dim])

        W1_steer = self.variable([state_dim,layer1_size],state_dim, name="steer_w1")
        b1_steer = self.variable([layer1_size],state_dim, name="steer_b1")
        W2_steer = self.variable([layer1_size,layer2_size],layer1_size, name="steer_w2")
        b2_steer = self.variable([layer2_size],layer1_size, name="steer_b2")

        W1_accel = self.variable([state_dim,layer1_size],state_dim, name="accel_w1")
        b1_accel = self.variable([layer1_size],state_dim, name="accel_b1")
        W2_accel = self.variable([layer1_size,layer2_size],layer1_size, name="accel_w2")
        b2_accel = self.variable([layer2_size],layer1_size, name="accel_b2")


#        W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
#        b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

        W3_steer = tf.Variable(tf.random_uniform([layer2_size,1],-1e-4,1e-4), name="steer_w3")
        b3_steer = tf.Variable(tf.random_uniform([1],-1e-4,1e-4), name="steer_b3")

        W3_accel = tf.Variable(tf.random_uniform([layer2_size,1],-1e-4,1e-4), name="accel_w3")
        b3_accel = tf.Variable(tf.random_uniform([1],-1e-4,1e-4), name="accel_b3")

        W_brake = tf.Variable(tf.random_uniform([layer2_size,1],-1e-4,1e-4), name="brake_w3")
        b_brake = tf.Variable(tf.random_uniform([1],-1e-4,1e-4), name="brake_b3")

        
        layer1_steer = tf.nn.relu(tf.matmul(state_input,W1_steer) + b1_steer)
        layer2_steer = tf.nn.relu(tf.matmul(layer1_steer,W2_steer) + b2_steer)

        layer1_accel = tf.nn.relu(tf.matmul(state_input, W1_accel) + b1_accel)
        layer2_accel = tf.nn.relu(tf.matmul(layer1_accel, W2_accel) + b2_accel)

        self.accel = tf.matmul(layer2_accel, W3_accel) + b3_accel
        self.brake = tf.matmul(layer2_accel, W_brake) + b_brake
        self.steer = tf.matmul(layer2_steer, W3_steer) + b3_steer
        steer = tf.tanh(tf.matmul(layer2_steer,W3_steer) + b3_steer)
        accel = tf.sigmoid(tf.matmul(layer2_accel,W3_accel) + b3_accel)
        brake = tf.sigmoid(tf.matmul(layer2_accel,W_brake) + b_brake)
#        print(layer2_accel, W_brake, b_brake)
        # action_output = tf.concat(1, [steer, accel, brake])
        #print([W1_steer,b1_steer,W2_steer,b2_steer,W3_steer,b3_steer,W1_accel,b1_accel,W2_accel,b2_accel,W3_accel,b3_accel,W_brake,b_brake])
        #print(tf.trainable_variables())
        action_output = tf.concat([steer, accel, brake], 1)

        return layer2_accel,state_input,action_output, tf.trainable_variables()#[W1_steer,b1_steer,W2_steer,b2_steer,W3_steer,b3_steer,W1_accel,b1_accel,W2_accel,b2_accel,W3_accel,b3_accel,W_brake,b_brake]

    def create_target_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]
        #print([x.name for x in tf.trainable_variables()])
        layer1_steer = tf.nn.relu(tf.matmul(state_input,self.net[0]) + self.net[1])
        layer2_steer = tf.nn.relu(tf.matmul(layer1_steer,self.net[2]) + self.net[3])

        layer1_accel = tf.nn.relu(tf.matmul(state_input, self.net[4]) + self.net[5])
        layer2_accel = tf.nn.relu(tf.matmul(layer1_accel, self.net[6]) + self.net[7])

        
        steer = tf.tanh(tf.matmul(layer2_steer,self.net[8]) + self.net[9])
        accel = tf.sigmoid(tf.matmul(layer2_accel,self.net[10]) + self.net[11])
        brake = tf.sigmoid(tf.matmul(layer2_accel,self.net[12]) + self.net[13])

        # action_output = tf.concat(1, [steer, accel, brake])
        action_output = tf.concat([steer, accel, brake], 1)
        return state_input,action_output,target_update,target_net




    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch
            })

    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })

    def action(self,state):
        #print("Layer 2 Accel: ", self.sess.run(self.a, feed_dict={self.state_input:[state]}))
        #print("Weights: ", self.sess.run(self.trainable_weights, feed_dict={self.state_input:[state]}))
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:[state]
            })[0], self.sess.run([self.accel, self.brake], feed_dict={self.state_input:[state]})


    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch
            })

    # f fan-in size
    def variable(self,shape,f,name):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)), name=name)
    '''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

    '''


