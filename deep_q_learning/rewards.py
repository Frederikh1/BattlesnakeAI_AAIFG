# No code has been tested, boiler code from our friend. 

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


def is_low_health(game):
    my_snake = game["you"]
    health = my_snake["health"]
    return health < 25

def is_high_health(game):
    my_snake = game["you"]
    health = my_snake["health"]
    return health >= 25


def is_wall_collision(game):
    my_head = game["you"]["body"][0]
    board_width = game["board"]["width"]
    board_height = game["board"]["height"]

    return my_head["x"] < 0 or my_head["x"] >= board_width or my_head["y"] < 0 or my_head["y"] >= board_height


def is_self_collision(game):
    my_head = game["you"]["body"][0]
    my_body = game["you"]["body"][1:]  # Exclude the head

    return my_head in my_body

