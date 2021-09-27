# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + list(permutations([i for i in range(m)], 2))
        self.state_space = state_space = [[a, b, c] for a in range(m) for b in range(t) for c in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod=[0 for _ in range(m+t+d)]
        state_encod[self.fetch_state_loc(state)]= 1
        state_encod[m+self.fetch_state_time(state)]= 1
        state_encod[m+t+self.fetch_state_day(state)]= 1

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] 
        actions = [self.action_space[i] for i in possible_actions_index]

        
        #actions.append([0,0])

        return possible_actions_index, actions   



    def reward_func(self, wait_time, transit_time, ride_time):
        """Takes in state, wait time transit time and ride time and returns the reward"""
        # transit and wait time yield no revenue, only battery costs, so they are idle times.
        idle_time= wait_time + transit_time
        
        reward= (R * ride_time) - (C - (ride_time + idle_time))
        
        return reward



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state=[]
        
        # Initialize various times
        total_time   = 0
        transit_time = 0    # to go from current  location to pickup location
        wait_time    = 0    # in case driver chooses to refuse all requests
        ride_time    = 0    # from Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc = self.fetch_state_loc(state)
        pickup_pos = self.get_pickup(action)
        drop_pos = self.get_drop(action)
        curr_time = self.fetch_state_time(state)
        curr_day = self.fetch_state_day(state)
        """
         3 Scenarios: 
           a) Refuse all requests
           b) Driver is already at pick up point
           c) Driver is not at the pickup point.
        """
        if((pickup_pos==0) and (drop_pos==0)):
                              
            # Refuse all requests, so wait time is 1 unit, next location is current location
            wait_time= 1
            next_loc= curr_loc
        
        elif(curr_loc==pickup_pos):
            
            # means driver is already at pickup point, wait and transit are both 0 then.
            ride_time=Time_matrix[curr_loc][drop_pos][curr_time][curr_day]
            
            # next location is the drop location
            next_loc=drop_pos
        
        else:
            
            # Driver need to go to pickup point, he needs to travel to pickup point first
            # time take to reach pickup point
            transit_time= Time_matrix[curr_loc][pickup_pos][curr_time][curr_day]
            new_time, new_day=self.get_ride_day_time(curr_time, curr_day, transit_time)
            
            ride_time=Time_matrix[pickup_pos][drop_pos][curr_time][curr_day]
            next_loc  = drop_pos
        
        # Calculate total time as sum of all durations
        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.get_ride_day_time(curr_time, curr_day, total_time)
        
        # So the next_state will be using the next_loc and the new time states.
        next_state = [next_loc, next_time, next_day]
            
        
        return next_state, wait_time, transit_time, ride_time




    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        # fetch the next state and the compute for time and day after the trip
        next_state, wait_time, transit_time, ride_time=self.next_state_func(state, action, Time_matrix)
        
        # Calculate the reward based on the different time durations
        rewards=self.reward_func(wait_time, transit_time, ride_time)
        total_time = wait_time + transit_time + ride_time
        
        return rewards, next_state, total_time
        
    
    def get_ride_day_time(self, time, day, time_taken):
        new_time = time + math.ceil(time_taken)
        day_of_week = day
        
        if new_time > 23:
            new_time = new_time % 24
            day_of_week += 1
            if day_of_week > 6:
                day_of_week = day_of_week % 7
        
        
        return new_time,day_of_week
    
    def fetch_state_loc(self,state):
        return state[0]
    
    def fetch_state_time(self, state):
        return state[1]
    
    def fetch_state_day(self, state):
        return state[2]
    
    def get_pickup(self, action):
        return action[0]
    
    def get_drop(self, action):
        return action[1]
        
