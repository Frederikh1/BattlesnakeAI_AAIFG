import random
from collections import namedtuple
from deep_q_learning.agent import Agent, Direction
import deep_q_learning.state as st
import deep_q_learning.rewards as rw
import deep_q_learning.save_statistics as save
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
    id = st.get_id_from_game(game)
    if(not (id in self.current_games)):
      return
    game_over = True
    reward = self.get_reward(game)
    score = game["you"]["length"]
    action = self.agent.get_next_move(game, reward, game_over, score)
    self.agent.done(game)
    save.save_stats(game, rw.isSnakeAlive(game))
    del self.current_games[id]

  def play_step(self, game):
    if(not rw.isSnakeAlive(game)):
      self.end(game)
      return
    reward = self.get_reward(game)
    game_over = False
    score = game["you"]["length"]
    action = self.agent.get_next_move(game, reward, game_over, score)
    self.save_game(game)
    return action
  
  def get_reward(self, game):
    reward = 0

    #Conditions
    reward += rw.did_mySnake_win(game)
    reward += 0 if rw.preserve_health(game) else -1
    reward += 2 if rw.is_low_health(game) and self.is_food_consumed(game) else 0
    reward += -5 if rw.is_wall_collision(game) else 0
    reward += -5 if rw.is_self_collision(game) else 0
    reward += -2 if rw.is_high_health(game) and self.is_food_consumed(game) else 0
    return reward

  def is_food_consumed(self, game):
    snake_head = game["you"]["head"]
    food_positions = self.get_last_food_positions(game)
    has_eaten_food = False
    for food_pos in food_positions:
      if snake_head == food_pos:
        print("Food has been consumed")
        has_eaten_food = True
        break
    return has_eaten_food
  
  def get_last_food_positions(self, game):
    id = st.get_id_from_game(game)
    state = self.current_games[id]
    food_positions = state["board"]["food"]
    return food_positions
  
  def save_game(self, game):
    id = st.get_id_from_game(game)
    self.current_games[id] = game

