import random
from collections import namedtuple
from deep_q_learning.agent import Agent, Direction
import numpy as np


class SnakeGameAI:
  latestGame = None
  agent = Agent()
  game_over = False
  last_food_position = None

  def __init__(self):
    self.direction = Direction.UP

  def reset(self, game):
    # init game state
    self.last_food_position = game["board"]["food"]
    self.agent.start_position(game)

  def set_state(self, state):
    self.latestState = state

  def end(self, game):
    self.game_over = True
    reward = self.get_reward(game)
    score = game["you"]["length"]
    action = self.agent.get_next_move(game, reward, self.game_over, score)
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
    return reward

  def get_current_health(self, game):
    health = game["you"]["health"]
    return health

  def is_food_consumed(self, game):
    snake_head = game["you"]["head"]
    food_positions = self.last_food_position
    has_eaten_food = False
    for food_pos in food_positions:
      if snake_head == food_pos:
        print("Food has been consumed")
        has_eaten_food = True
        break
    self.last_food_position = game["board"]["food"]
    return has_eaten_food
    return False

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
