# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 10:25:00 2016

@author: s6324900
"""

import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q = {}
        self.epsilon = 0.1
        self.alpha = 0.5
        self.gamma = 0.9
        self.actions = ['forward','left','right', None]
        self.lastState = None
        self.lastAction = None        
        # TODO: Initialize any additional variables here
        action = random.choice(self.actions)
        self.q[(self.state, action)] = 2.0           
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        deadline = self.env.get_deadline(self) # For printing in the end
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)        
           
        
        # TODO: Select action according to your policy
        q_options = []
        def next_action(state):
        # Exploration using epsilon parameter
            if random.random() < self.epsilon:
                action = random.choice(self.actions)
        # Exploitation using q learning matrix to choose best action
            else:
                for a in self.actions:
                    q_options.append(self.q.get((self.state,a), 2.0))
                max_q = max(q_options)
        #Return action corresponding to Max Q
                new_dict = dict((v,k) for k,v in self.q.iteritems())
                best_tuple = new_dict[max_q]
                action = best_tuple[1]
            return action
            
        action = next_action(self.state)
        
        # Store current values of state and action
        self.lastState = self.state
        self.lastAction = action
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Update state once action is taken
        inputs_new = self.env.sense(self)  
        self.state = (inputs_new['light'], inputs_new['oncoming'], inputs_new['left'], inputs_new['right'], self.planner.next_waypoint)                
        # Calculate Max q new for the new state
        qnew_options = [] 
        for a in self.actions:
            qnew_options.append(self.q.get((self.state, a), 2.0))
        maxqnew = max(qnew_options)
        value = reward + (self.gamma*maxqnew) 
    
        #Update q table based on the reward
        curr_q = self.q.get((self.lastState, self.lastAction), None)
        if curr_q is None:
            self.q[(self.lastState, self.lastAction)] = reward
        else:
            self.q[(self.lastState, self.lastAction)] = curr_q + self.alpha * (value - curr_q)
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, next_waypoint = {}, action = {}, reward = {}".format(deadline, inputs, self.next_waypoint, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    #print number of successful trips
    print('\nSuccessful trips: ', e.get_success())

if __name__ == '__main__':
    run()
