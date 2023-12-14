class State:
    current_games = None
    
    def __init__(self):
        self.current_games = {}
    
    def __add_or_update_state(self, id, game):
        self.current_games[id] = game
    
    def __get_state(self, id):
        state = self.current_games[id]
        return state
    
    def clear(self, game):
        id = get_id_from_game(game)
        self.current_games.pop(id)
    
    def save_old_states(self, game, old_state, final_move):
        id = get_id_from_game(game)
        states = {"old_state": old_state, "final_move": final_move}
        self.__add_or_update_state(id, states)

    def get_old_states(self, game):
        id = get_id_from_game(game)
        states = self.__get_state(id)
        old_state = states["old_state"]
        final_move = states["final_move"]
        return old_state, final_move
    
def get_id_from_game(game):
  return game["game"]["id"]
    
    