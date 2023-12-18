import struct
import torch
import random
import numpy as np
import deep_q_learning.a_star as astar
from deep_q_learning.floodFill import flood_fill
from collections import deque
from deep_q_learning.model import Linear_QNet, QTrainer
from deep_q_learning.state import State
from enum import Enum


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Direction(Enum):
  DOWN = 0
  LEFT = 1
  UP = 2
  RIGHT = 3

class Agent:
  old_states = None
  current_direction = None

  def __init__(self):
    self.n_games = 0
    self.epsilon = 0  # randomness
    self.gamma = 0.9  # discount rate
    self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
    self.model = Linear_QNet(16, 256, 3)
    #comment if you don't want to load from the saved model
    self.model.load()
    print('loaded saved model')
    self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    self.old_states = State()

  def get_state(self, game):
    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game["you"]["body"][0]  # Coordinates of your head
    my_neck = game["you"]["body"][1]  # Coordinates of your "neck"
    food = game["board"]["food"]

    snake_board = self.get_snake_positions(game)
    collision_left = self.get_next_collision(my_head, [-1, 0], snake_board)
    collision_up = self.get_next_collision(my_head, [0, 1], snake_board)
    collision_right = self.get_next_collision(my_head, [1, 0], snake_board)
    collision_down = self.get_next_collision(my_head, [0, -1], snake_board)

    dir_l = False
    dir_r = False
    dir_u = False
    dir_d = False
    
    self.current_direction = self.get_direction(my_neck, my_head)
    direction_input = self.convert_to_bool_directions(self.current_direction)
    
    distance_to_food, food_path_coordinate = self.get_closest_food(food, my_head, snake_board)
    food_path_coordinate = self.direction_to_dictionary(food_path_coordinate)
    food_path_direction = self.get_direction(my_head, food_path_coordinate)
    food_path_direction_inputs = self.convert_to_bool_directions(food_path_direction)

    # Flood Fill -- Start --
    # Using Flood Fill to assess each potential move
    current_position = (my_head["x"], my_head["y"])

    # Potential moves [left, straight, right]
    potential_moves = self.get_potential_moves(current_position, self.current_direction)
    move_scores = {}

    for move_direction, new_position in potential_moves.items():
      temp_board = np.copy(snake_board)  # Copy board to simulate the move
      accessible_area_size = flood_fill(temp_board, new_position)
      move_scores[move_direction] = accessible_area_size

    # Flood Fill -- End --

    state = [
        #The three dangers are currently placeholders to avoid get to work first
        # Danger straight
        move_scores["straight"],

        # Danger right
        move_scores["right"],

        # Danger left
        move_scores["left"],

        collision_left,
        collision_up,
        collision_right,
        collision_down,
        
        distance_to_food
    ]
    
    #Add move direction
    state+= direction_input
    
    #Adding directions to get to food the quickest
    state+=food_path_direction_inputs

    return np.array(state, dtype=int)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state,
                        done))  # popleft if MAX_MEMORY is reached

  def train_long_memory(self):
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
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
    final_move = [0, 0, 0]
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
    state_old = self.get_state(game)

    # get move
    final_move = self.get_action(state_old)
    
    self.old_states.save_old_states(game, state_old, final_move)
    
    self.current_direction = Direction.UP

  def get_next_move(self, game, reward, done, score):
    # perform move and get new state
    state_new = self.get_state(game)
    state_old, final_move = self.old_states.get_old_states(game)
    # train short memory
    self.train_short_memory(state_old, final_move, reward, state_new, done)
    
    # remember
    self.remember(state_old, final_move, reward, state_new, done)

    # get old state
    state_old = state_new

    # get move
    final_move = self.get_action(state_old)
    self.old_states.save_old_states(game, state_old, final_move)
    return self.__direction_in_string(final_move)
  
  def get_potential_moves(self, current_position, current_direction):
    # Mapping from current direction to new direction after a turn
    turn_mapping = {
        Direction.UP: [Direction.LEFT, Direction.UP, Direction.RIGHT],
        Direction.RIGHT: [Direction.UP, Direction.RIGHT, Direction.DOWN],
        Direction.DOWN: [Direction.RIGHT, Direction.DOWN, Direction.LEFT],
        Direction.LEFT: [Direction.DOWN, Direction.LEFT, Direction.UP]
    }

    # Get possible new directions
    possible_directions = turn_mapping[current_direction]

    # Calculate new head positions for each possible direction
    move_positions = {
        'left': self.get_new_position(current_position, possible_directions[0]),
        'straight': self.get_new_position(current_position, possible_directions[1]),
        'right': self.get_new_position(current_position, possible_directions[2])
    }
    return move_positions

  def get_new_position(self, position, direction):
    # Calculate new position based on direction
    if direction == Direction.UP:
        return (position[0], position[1] + 1)
    elif direction == Direction.RIGHT:
        return (position[0] + 1, position[1])
    elif direction == Direction.DOWN:
        return (position[0], position[1] - 1)
    elif direction == Direction.LEFT:
        return (position[0] - 1, position[1])

  def convert_direction_to_move(self, direction):
    # Convert direction to final_move format
    if direction == 'left':
        return [1, 0, 0]
    elif direction == 'straight':
        return [0, 1, 0]
    elif direction == 'right':
        return [0, 0, 1]

  def done(self):
    self.n_games += 1
    self.train_long_memory()
    self.model.save()

  def __direction_in_string(self, direction_state):
    direction = 0
    if direction_state == [1, 0, 0]:  #left
      direction = -1
    elif direction_state == [0, 0, 1]:  #right
      direction = +1

    new_direction = (self.current_direction.value + direction) % len(Direction)
    direction = (self.get_direction_from_value(new_direction))
    self.current_direction = direction
    return direction.name.lower()

  def get_direction_from_value(self, value):
    for direction in Direction:
      if direction.value == value:
        return direction

  def get_next_collision(self, head, direction, board):
    min = 0
    start = min + 1
    max = len(board) - 1
    next_collision = 0
    if head["x"] > max or head["y"] > max:
      return next_collision
    if (direction[0] != 0):
      head_x = head["x"]
      for x in range(start, max):
        position = head_x + (x * direction[0])
        if (position < min or position > max
            or board[position][head["y"]] != 0):
          break
        next_collision += 1
    if (direction[1] != 0):
      head_y = head["y"]
      for x in range(start, max):
        position = head_y + (x * direction[0])
        if (position < min or position > max
            or board[head["x"]][position] != 0):
          break
        next_collision += 1
    return next_collision
  
  def get_closest_food(self, food_positions, head, snake_board):
    head = (head["x"], head["y"])
    max_range = 11*11
    closest_food = max_range
    coordinate = None
    for position in food_positions:
      goal = (position["x"], position["y"])
      path = astar.astar(snake_board, head, goal)
      if path and len(path)<closest_food:
        closest_food = len(path)
        coordinate = [path[1][0], path[1][1]]
    return closest_food, coordinate
      

  def get_snake_positions(self, game):
    board = self.create_board()
    snakes = game["board"]["snakes"]
    own_snake = game["you"]

    self.add_positions(own_snake["body"], board, 1)

    identifier = 2
    for snake in snakes:
      if (own_snake["id"] == snake["id"]):
        continue

      body = snake["body"]
      self.add_positions(body, board, identifier)
      identifier += 1

    return board

  def create_board(self):
    dimension = 11
    board = np.zeros((dimension, dimension), dtype=int)
    return board

  def add_positions(self, positions, board, value):
    for x in positions:
      if x["x"] > 10 or x["y"] > 10:
        continue
      board[x["x"]][x["y"]] = value
      
  def direction_to_dictionary(self, array):
    if array is None:
      return
    dictionary = {"x": array[0], "y": array[1]}
    return dictionary
      
  def get_direction(self, current, goal):
    if current is None or goal is None:
        return None 
    if current["x"] < goal["x"]:
      return Direction.RIGHT

    elif current["x"] > goal["x"]:
      return Direction.LEFT

    elif current["y"] < goal["y"]:
      return Direction.UP

    elif current["y"] > goal["y"]:
      return Direction.DOWN
    
    return Direction.UP
    
  #This should always put the directions in the same spot
  def convert_to_bool_directions(self, direction):
    directions = []
    for dir in Direction:
      directions.append(False)
    
    if(direction is not None):
      directions[direction.value] = True
    return directions
      