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
    
    def get_reward(self, game):
        reward = 0
        reward += self.is_mySnake_alive(game)
        if(self.is_food_consumed(game)):
            reward += 2
        return 0

    def get_current_health(self, game):
        health = game["you"]["health"]
        return health

    def is_food_consumed(self, game):
        snake_head = game["you"]["head"]
        food_positions = game["board"]["food"]
        for food_pos in food_positions:
            if snake_head == food_pos:
                print("Food has been consumed")
                break

    def is_mySnake_alive(self, game):
        my_snake_id = game["you"]["id"]
        currently_alive_snakes = game["board"]["snakes"]

        for snake in currently_alive_snakes:
            if snake["id"] == my_snake_id:
                print("Alive")
                break  
            else:
                print("Dead")
