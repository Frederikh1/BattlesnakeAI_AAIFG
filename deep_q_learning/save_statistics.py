import csv
import os

base_length = 3

def save_stats(game, isWin):
    statistics = []
    turn = game["turn"]
    food_consumed = game["you"]["length"] - base_length
    health = game["you"]["health"]
    isWin = isWin > 0
    statistics.extend([turn, food_consumed, health, str(isWin)])
    __create_or_append_csv(statistics)

def __create_or_append_csv(statistics):
    performance_folder_path = './performance'
    file_name = "performance.csv"
    with_header = os.path.isfile(performance_folder_path + "/" + file_name)
    if not os.path.exists(performance_folder_path):
        os.makedirs(performance_folder_path)

    with open(performance_folder_path + "/" + file_name, "a", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        if(not with_header):
            writer.writerow(get_header())
        writer.writerow(statistics)

def get_header():
    return ["turn", "food_consumed", "health", "win"]