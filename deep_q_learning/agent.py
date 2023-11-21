import torch
import random
import numpy as np
from collections import deque
from .model import Linear_QNet, QTrainer
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
            (dir_r and self.is_collision(Direction.RIGHT)) or #my_neck["x"] < my_head["x"]
            (dir_l and self.is_collision(Direction.LEFT)) or #my_neck["x"] > my_head["x"] 
            (dir_u and self.is_collision(Direction.UP)) or #my_neck["y"] < my_head["y"] 
            (dir_d and self.is_collision(Direction.DOWN)), #my_neck["y"] > my_head["y"]

            # Danger right
            (dir_r and self.is_collision(Direction.RIGHT)) or 
            (dir_l and self.is_collision(Direction.LEFT)) or 
            (dir_u and self.is_collision(Direction.UP)) or 
            (dir_d and self.is_collision(Direction.DOWN)),

            # Danger left
            (dir_r and self.is_collision(Direction.RIGHT)) or 
            (dir_l and self.is_collision(Direction.LEFT)) or 
            (dir_u and self.is_collision(Direction.UP)) or 
            (dir_d and self.is_collision(Direction.DOWN)),
            
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





    #function that determines if there is a DANGEROUS collision after moving in point DIRECTION
    def is_collision(self, point):
        me = game["you"]["id"]            # my snake id
        my_head = game["you"]["body"][0]  # Coordinates of your head
        my_neck = game["you"]["body"][1]  # Coordinates of your "neck"
        board_height = game["board"]["height"] 
        board_width = game["board"]["width"] 
        snakes = game["board"]["snakes"]  #array of battlesnakes remaining on the game board
        #only useful in certain game modes (check battlesnake API):
        #hazards = game["board"]["hazards"] # to get the list of coords of the hazards.

        #TODO: check what happens on the next move (if we go on a certain direction),
        # not the current situation.
        # Is there a way to simulate the move, so I can check the simulated future situation?
        # hits boundary
        if my_head['x'] > board_width or
            my_head['x'] < 0 or my_head['y'] > board_height or my_head['y'] < 0:
            return True

        # hits itself
        if my_head['x'] in my_neck[1:]:
            return True

        #hits other snakes
        for i in len(snakes):
            if snakes[i]["id"] != me: # considers only other snakes
                its_head = snakes[i]["head"]        #coordinates of this snake's head
                its_body = snakes[i]["body"][1:]    #coordinates of this snake's body (head excluded)
                if snake_body_collision(its_head, its_body, my_head, my_neck):
                    return True
                if snake_head_collision(its_head, its_body, my_head, my_neck) == 'bad':
                    return True
                return True
        return False

    #return True if collision happens, False if not
    def snake_body_collision(its_head, its_body, my_head, my_neck):
        for i, body in enumerate(its_body):
            if my_head['x'] == body['x'] and head['y'] == body['y']:
                return True
        return False

    #return 'bad' if the snake dies, 'good' if it survives
    def snake_head_collision(its_head, its_body, my_head, my_neck):
        if my_head['x'] == its_head['x'] and my_head['y'] == its_head['y']:
            if len(my_neck) <= len(its_body):
                return 'bad'
        return 'good'


    #TODO change is_collision into closest_collision: it should find the closest obstacle
    #and returns its coordinates (?)
    #Keep danger straight, right and left but it still needs the direction the snake is moving to
    #-->Or change to up,down,left,right so we know where the closest danger in each direction is


    #WIP: function to find closest piece of food to snake head
    #Can be implemented with different algorithms:
    #- Hamming distance
    #- Euclidian distance
    #- manhattan distance (taxicab or city bllock)
    #- Minkowsky distance
    def closest_food(algorithm="hamming"):
        foodies = game["board"]["food"] #food ccordinates array
        if algorithm="hamming":
            print("hamming distance")
            #TODO
        elif algorithm="euclidian":
            print("euclidian distance")
            #TODO
        elif algorithm="manhattan":
            print("manhattan distance")
            #TODO
        elif algorithm="minkowski":
            print("minkowski distance")
            #TODO






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
        
        
        