import torch
import random
import numpy as np
from collections import deque
from deep_q_learning.model import Linear_QNet, QTrainer
from enum import Enum

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Direction(Enum):
    DOWN = 1
    LEFT = 2
    UP = 3
    RIGHT = 4

class Agent:
    state_old = None
    final_move = None
    current_direction = None

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):       
        # We've included code to prevent your Battlesnake from moving backwards
        my_head = game["you"]["body"][0]  # Coordinates of your head
        my_neck = game["you"]["body"][1]  # Coordinates of your "neck"
        food = game["board"]["food"][0]
        
        dir_l = False
        dir_r = False
        dir_u = False
        dir_d = False
        
        # We probably need some kind of vector as basis figure out which locations in the field which are unreachable
        if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
            dir_l = True
            self.current_direction = Direction.RIGHT

        elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
            dir_r = True
            self.current_direction = Direction.LEFT

        elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
            dir_u = True
            self.current_direction = Direction.UP

        elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
            dir_d = True
            self.current_direction = Direction.DOWN

        state = [
            #The three dangers are currently placeholders to avoid get to work first
            # Danger straight
            (dir_r and my_neck["x"] < my_head["x"]) or 
            (dir_l and my_neck["x"] > my_head["x"]) or 
            (dir_u and my_neck["y"] < my_head["y"]) or 
            (dir_d and my_neck["y"] > my_head["y"]),

            # Danger right
            (dir_r and my_neck["x"] < my_head["x"]) or 
            (dir_l and my_neck["x"] > my_head["x"]) or 
            (dir_u and my_neck["y"] < my_head["y"]) or 
            (dir_d and my_neck["y"] > my_head["y"]),

            # Danger left
            (dir_r and my_neck["x"] < my_head["x"]) or 
            (dir_l and my_neck["x"] > my_head["x"]) or 
            (dir_u and my_neck["y"] < my_head["y"]) or 
            (dir_d and my_neck["y"] > my_head["y"]),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            food['x'] < my_head['x'],  # food left
            food['x'] > my_head['x'],  # food right
            food['y'] < my_head['y'],  # food up
            food['y'] > my_head['y']  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    def start_position(self, game):
        # get old state
        self.state_old = self.get_state(game)

        # get move
        self.final_move = self.get_action(self.state_old)
        self.current_direction = Direction.UP
    
    def get_next_move(self, game, reward, done, score):
        # perform move and get new state
        state_new = self.get_state(game)

        # train short memory
        self.train_short_memory(self.state_old, self.final_move, reward, state_new, done)

        # remember
        self.remember(self.state_old, self.final_move, reward, state_new, done)
                
        # get old state
        self.state_old = state_new

        # get move
        self.final_move = self.get_action(self.state_old)
        return self.__direction_in_string(self.final_move)
    
    def done(self):
        self.n_games += 1
        self.train_long_memory()
        self.model.save()
    
    def __direction_in_string(self, direction_state):
        print(direction_state)
        direction = 0
        if direction_state == [1, 0, 0]: #left
            direction =- 1
        elif direction_state == [0, 0, 1]: #right
            direction =+ 1
        
        new_direction = (self.current_direction.value + direction) % 4
        print(new_direction)
        direction = (self.get_direction_from_value(new_direction))
        print(direction)
        self.current_direction = direction
        return direction.name.lower()
            
    def get_direction_from_value(self, value):
        for direction in Direction:
            if direction.value == value:
                return direction
    
    def get_current_health(game):
        health = game["you"]["health"]
        return health
    
    def is_food_consumed(game):
        snake_head = game["you"]["head"]
        food_positions = game["board"]["food"]
        for food_pos in food_positions:
            if snake_head == food_pos:
                print("Food has been consumed")
                break

    def is_mySnake_alive(game):
        my_snake_id = game["you"]["id"]
        currently_alive_snakes = game["board"]["snakes"]

        for snake in currently_alive_snakes:
            if snake["id"] == my_snake_id:
                print("Alive")
                break  
            else:
                print("Dead")

