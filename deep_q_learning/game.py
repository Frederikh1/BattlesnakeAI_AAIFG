import random
from collections import namedtuple
from deep_q_learning.agent import Agent, Direction
import numpy as np

class SnakeGameAI:
    latestGame = None
    agent = Agent()
    game_over = False

    def __init__(self):
        self.direction = Direction.UP

    def reset(self, game):
        # init game state
        self.agent.start_position(game)       
        
    def set_state(self, state):
        self.latestState = state

    def end(self):
        self.game_over = True
        self.agent.done()

    def play_step(self, game):
        reward = 0
        score = game["you"]["length"]
        action = self.agent.get_next_move(game, reward, self.game_over, score)
        return action
