import random
from collections import namedtuple
from deep_q_learning.agent import Agent, Direction
import deep_q_learning.state as st
import numpy as np


class SnakeGameAI:
  current_games = None
  agent = None

  def __init__(self):
    self.agent = Agent()
    self.current_games = {}

  def reset(self, game):
    # init game state
    self.save_game(game)
    self.agent.start_position(game)

  def set_state(self, state):
    self.latestState = state

  def end(self, game):
    game_over = True
    reward = self.get_reward(game)
    score = game["you"]["length"]
    action = self.agent.get_next_move(game, reward, game_over, score)
    self.agent.done()
  
  def play_step(self, game):
    reward = self.get_reward(game)
    score = game["you"]["length"]
    action = self.agent.get_next_move(game, reward, self.game_over, score)
    return action
  
  def get_reward(self, game):
    reward = 0
    reward += self.did_mySnake_win(game)
    if (self.is_food_consumed(game)):
        reward += 1
    elif (self.get_current_health(game) < 20 & self.is_food_consumed(game) == True):
        reward += 2
    return reward

  def get_current_health(self, game):
    health = game["you"]["health"]
    return health

  def is_food_consumed(self, game):
    snake_head = game["you"]["head"]
    food_positions = self.get_last_food_positions(game)
    has_eaten_food = False
    for food_pos in food_positions:
      if snake_head == food_pos:
        print("Food has been consumed")
        has_eaten_food = True
        break
    self.save_game(game)
    return has_eaten_food

  def did_mySnake_win(self, game):
    my_snake_id = game["you"]["id"]
    currently_alive_snakes = game["board"]["snakes"]
    snake_ids = [snake["id"] for snake in currently_alive_snakes]
    my_snake_count = snake_ids.count(my_snake_id)
    # my_snake_index = snake_ids.index(my_snake_id) if my_snake_id in snake_ids else -1

    is_unique_last_snake = my_snake_count == 1 and len(snake_ids) == 1

    if is_unique_last_snake:
      print("-- Won --")
      return 10
    elif len(snake_ids) > 1:
      return 0
    print("-- Lost --")
    return -10
  
  def get_last_food_positions(self, game):
    id = st.get_id_from_game(game)
    state = self.current_games[id]
    food_positions = state["board"]["food"]
    return food_positions
  
  def save_game(self, game):
    id = st.get_id_from_game(game)
    self.current_games[id] = game
