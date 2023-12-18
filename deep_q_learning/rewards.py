# No code has been tested, boiler code from our friend. 
def get_current_health(game):
    health = game["you"]["health"]
    return health

def did_mySnake_win(game):
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


def is_head_to_head_win(game):
    my_snake = game["you"]
    my_length = len(my_snake["length"])

    for snake in game["board"]["snakes"]:
        if snake["id"] != my_snake["id"] and len(snake["body"]) < my_length:
            if my_snake["body"][0] == snake["body"][0]:
                print("Head to head win!")
                return True
    print("Head to head Loss!")
    return False


def is_consuming_food_low_health(game):
    my_snake = game["you"]
    health = my_snake["health"]
    food = game["board"]["food"]

    # Assuming you have a way to check if food was consumed this turn
    return health < 25 and my_snake["body"][0] in food


def is_taking_space(game):
    # Implement logic to check if the snake is taking space effectively
    pass

def is_taking_space(game):
    # Assuming you have a flood_fill function and a way to create a board representation
    board = create_board_representation(game)
    start = (game["you"]["body"][0]["x"], game["you"]["body"][0]["y"])
    area_size = flood_fill(board, start)

    # Define what you consider "effectively taking space"
    return area_size > some_threshold_value

def is_consuming_food_high_health(game):
    my_snake = game["you"]
    health = my_snake["health"]
    food = game["board"]["food"]

    # Assuming you have a way to check if food was consumed this turn
    return health >= 25 and my_snake["body"][0] in food


def is_wall_collision(game):
    my_head = game["you"]["body"][0]
    board_width = game["board"]["width"]
    board_height = game["board"]["height"]

    return my_head["x"] < 0 or my_head["x"] >= board_width or my_head["y"] < 0 or my_head["y"] >= board_height


def is_self_collision(game):
    my_head = game["you"]["body"][0]
    my_body = game["you"]["body"][1:]  # Exclude the head

    return my_head in my_body

def preserve_health(game):
    health = game["you"]["health"]
    if (health >= 15):
        return True
    else:
        return False
    
def isSnakeAlive(game):
    my_snake_id = game["you"]["id"]
    game_id = game["game"]["id"]
    currently_alive_snakes = game["board"]["snakes"]

    is_alive = any(snake["id"] == my_snake_id for snake in currently_alive_snakes)

    if is_alive:
        return True
    else:
        return False