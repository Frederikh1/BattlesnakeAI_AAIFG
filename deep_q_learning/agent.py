import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    state_old = None
    final_move = None

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

        elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
            dir_r = True

        elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
            dir_u = True

        elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
            dir_d = True

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
    
    def get_action(self, game):
        # perform move and get new state
        reward, done, score = game.play_step(self.final_move)
        state_new = self.get_state(game)

        # train short memory
        self.train_short_memory(self.state_old, self.final_move, reward, state_new, done)

        # remember
        self.remember(self.state_old, self.final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            self.n_games += 1
            self.train_long_memory()

            if score > record:
                record = score
                self.model.save()
                
            # get old state
        self.state_old = self.get_state(game)

        # get move
        self.final_move = self.get_action(self.state_old)
        return self.final_move
    
    def done(self):
        self.n_games += 1
        self.train_long_memory()
        self.model.save()
    
    def train(self, game):
        return 0
        