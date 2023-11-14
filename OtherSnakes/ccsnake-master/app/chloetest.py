import bottle
import os

ENEMY = 1
WALL = 2
SAFE = 5

game_id = ''
board_width = 0
board_height = 0
board = None

@bottle.route('/')
def static():
    return "the server is running"

@bottle.route('/static/<path:path>')
def static(path):
    return bottle.static_file(path, root='static/')

@bottle.post('/start')
def start():
    data = bottle.request.json
    global game_id
    global board_width
    global board_height
    game_id = data.get('game_id')
    board_width = data.get('width')
    board_height = data.get('height')

    head_url = '%s://%s/static/head2.png' % (
        bottle.request.urlparts.scheme,
        bottle.request.urlparts.netloc
    )

    return {
        'color': '#E74C3C',
        'taunt': '{} ({}x{})'.format(game_id, board_width, board_height),
        'head_url': head_url,
        'name': 'cc',
        "head_type": "tongue",
        "tail_type": "pixel"
    }

@bottle.post('/move')
def move():
    data = bottle.request.json
    global board
    board = [[0 for x in range(board_width)] for y in range(board_height)]

    my_food_list = []
    food_list = data.get('food')['data']
    for each_food in food_list:
        food_x = each_food['y']
        food_y = each_food['x']
        my_food_list.append([food_x, food_y])

    my_body_list = []
    body_list = data.get('you')['body']['data']
    my_len = data.get('you')['length']
    my_id = data.get('you')['id']
    for each_segment in body_list:
        segment_x = each_segment['y']
        segment_y = each_segment['x']
        my_body_list.append([segment_x, segment_y])
    my_body_list.append(my_len)

    enemy_body_list = []
    enemy_list = data.get('snakes')['data']
    for each_enemy in enemy_list:
        each_enemy_snake = []
        each_enemy_len = each_enemy['length']
        each_enemy_health = each_enemy['health']
        print('enemy health', each_enemy_health)

        if (each_enemy['id'] != my_id) and (each_enemy_health != 0):
            for each_enemy_segement in each_enemy['body']['data']:
                enemy_body_x = each_enemy_segement['y']
                enemy_body_y = each_enemy_segement['x']
                each_enemy_snake.append([enemy_body_x, enemy_body_y])
            each_enemy_snake.append(each_enemy_len)
            enemy_body_list.append(each_enemy_snake)
    print(enemy_body_list)

    board = set_walls(my_food_list, my_body_list, enemy_body_list)
    head = my_body_list[0]
    directions = direction_options(head)
    direction = find_best_direction(directions, head, my_food_list)

    for i in board:
        print(i)
    return {
        'move': direction,
        'taunt': 'I\'m drunk'
    }

def set_walls(my_food_list, my_body_list, enemy_body_list):
    global board
    for each_segment in my_body_list[0:len(my_body_list) - 1:1]:
        board[each_segment[0]][each_segment[1]] = 1

    for each_snake in enemy_body_list:
        for each_segment in each_snake[0:len(each_snake) - 1:1]:
            board[each_segment[0]][each_segment[1]] = 1

    reopen = get_open_coordinates(my_food_list, my_body_list, enemy_body_list)
    for each_coor in reopen:
        board[each_coor[0]][each_coor[1]] = 0

    for each_enemy in enemy_body_list:
        if each_enemy[-1] >= my_body_list[-1]:
            enemy_head = each_enemy[0]
            head_next_location = next_move_location(enemy_head)
            for next in head_next_location:
                board[next[0]][next[1]] = 1

    return board

def next_move_location(current_location):
    global board
    next_location = []
    directions = direction_options(current_location)
    for next_move in directions:
        if next_move == 'up':
            next_location.append([current_location[0] - 1, current_location[1]])
        elif next_move == 'down':
            next_location.append([current_location[0] + 1, current_location[1]])
        elif next_move == 'left':
            next_location.append([current_location[0], current_location[1] - 1])
        elif next_move == 'right':
            next_location.append([current_location[0], current_location[1] + 1])
    return next_location

def direction_options(head):
    global board
    global board_width
    global board_height
    curx = head[1]
    cury = head[0]
    directions = []

    if cury >= 1:
        if board[cury - 1][curx] == 0:
            directions.append('up')
    if curx <= board_width - 2:
        if board[cury][curx + 1] == 0:
            directions.append('right')
    if curx >= 1:
        if board[cury][curx - 1] == 0:
            directions.append('left')
    if cury <= board_height - 2:
        if board[cury + 1][curx] == 0:
            directions.append('down')
    return directions

def find_best_direction(directions, head, my_food_list):
    direction = ''
    food_pos = my_food_list[0]
    min_distance_food = get_distance(head[1], head[0], my_food_list[0][1], my_food_list[0][0])

    for i in my_food_list:
        distance_food = get_distance(head[1], head[0], i[1], i[0])
        if distance_food <= min_distance_food:
            food_pos = i
            min_distance_food = distance_food

    headx = head[0]
    heady = head[1]
    foodx = food_pos[0]
    foody = food_pos[1]
    updistance, rightdistance, downdistance, leftdistance = float("inf"), float("inf"), float("inf"), float("inf")

    for legal_direction in directions:
        if legal_direction == 'up':
            updistance = get_distance(headx - 1, heady, foodx, foody)
        elif legal_direction == 'right':
            rightdistance = get_distance(headx, heady + 1, foodx, foody)
        elif legal_direction == 'left':
            leftdistance = get_distance(headx, heady - 1, foodx, foody)
        elif legal_direction == 'down':
            downdistance = get_distance(headx + 1, heady, foodx, foody)

    direction = 'up'
    min_distance = updistance
    if rightdistance < min_distance:
        direction = 'right'
        min_distance = rightdistance
    if downdistance < min_distance:
        direction = 'down'
        min_distance = downdistance
    if leftdistance < min_distance:
        direction = 'left'
        min_distance = leftdistance

    return direction

def get_open_coordinates(my_food_list, my_body_list, enemy_body_list):
    open_coor = []

    if my_body_list[-2] != my_body_list[-3]:
        my_next_locations = next_move_location(my_body_list[0])
        common = 0
        for i in my_next_locations:
            if i in my_food_list:
                common += 1
        if common == 0:
            open_coor.append(my_body_list[-2])

    for enemy_snake in enemy_body_list:
        if enemy_snake[-2] != enemy_snake[-3]:
            enemy_next_locations = next_move_location(enemy_snake[0])
            common = 0
            for i in enemy_next_locations:
                if i in my_food_list:
                    common += 1
            if common == 0:
                open_coor.append(enemy_snake[-2])

    return open_coor

def get_distance(x1, y1, x2, y2):
    distance = ((x1 - x2) ** 2) + ((y1 - y2) ** 2)
    return distance

@bottle.post('/end')
def end():
    data = bottle.request.json
    return {'taunt': 'uh'}

application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '7070'),
        debug=True
    )