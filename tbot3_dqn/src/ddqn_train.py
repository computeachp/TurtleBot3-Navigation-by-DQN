#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from math import pi
import os
import json
import sys
import time
import random
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray
from collections import deque
from env import Environment
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop, Adam
from keras.layers.core import Dense, Dropout, Activation

EPISODES = 3000

class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('tbot3_dqn/src', 'tbot3_dqn/src/save_model/ddqn_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 500
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath + str(self.load_episode) + ".h5").get_weights())

            with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            state = mini_batch[i][0]
            action = mini_batch[i][1]
            reward = mini_batch[i][2]
            next_state = mini_batch[i][3]
            done = mini_batch[i][4]

            X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)

            st_predict = self.q_value = self.model.predict(state.reshape(1, len(state)))
            nst_predict = self.model.predict(next_state.reshape(1, len(next_state)))
            if target:
                nst_predict_target = self.target_model.predict(next_state.reshape(1, len(next_state)))
            else:
                nst_predict_target = self.model.predict(next_state.reshape(1, len(next_state)))

            target = reward
            if not done:
                target = reward + self.discount_factor * nst_predict_target[0][np.argmax(nst_predict[0])]

            target_f = st_predict[0]
            target_f[action] = target

            Y_batch = np.append(Y_batch, np.array([target_f]), axis=0)


        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)

if __name__ == "__main__":
    rospy.init_node('tbot3_dqn')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    
    state_size = 28
    action_size = 5

    env = Environment(action_size)
    
    agent = DDQNAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    done_check = False
    
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        
        for t in range(agent.episode_step):
            action = agent.getAction(state)
            next_state, reward, done = env.step(action)
            agent.appendMemory(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if done_check:
                    agent.trainModel(True)
                else:
                    agent.trainModel()
            
            score += reward
            state = next_state
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)
            
            if t >= 500:
                rospy.loginfo("Time out.")
                done = True
            
            if e % 10 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)
            
            if done:
                done_check = True
                agent.updateTargetModel()
                rospy.loginfo("UPDATE TARGET NETWORK")
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                
                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d', e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))
                break
            
            global_step += 1
            # if global_step % agent.target_update == 0:
            #     done_check = True
            #     agent.updateTargetModel()
            #     rospy.loginfo("UPDATE TARGET NETWORK")
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay