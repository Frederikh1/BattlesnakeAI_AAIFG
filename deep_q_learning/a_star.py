import heapq
import time

start_time = time.time()

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = 0  # Actual cost from start to this node
        self.h = 0  # Heuristic cost from this node to goal
        self.parent = None

    def __lt__(self, other):
        # Comparison function for priority queue
        return (self.g + self.h) < (other.g + other.h)

def heuristic(node, goal):
    # Calculate the Manhattan distance heuristic
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def astar(grid, start, goal):
    open_list = []
    closed_set = set()

    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Four possible moves (up, down, left, right)

        for dx, dy in neighbors:
            neighbor_x, neighbor_y = current_node.x + dx, current_node.y + dy

            if (
                0 <= neighbor_x < len(grid)
                and 0 <= neighbor_y < len(grid[0])
                and grid[neighbor_x][neighbor_y] < 1
                and (neighbor_x, neighbor_y) not in closed_set
            ):
                neighbor_node = Node(neighbor_x, neighbor_y)
                neighbor_node.g = current_node.g + 1
                neighbor_node.h = heuristic(neighbor_node, goal_node)
                neighbor_node.parent = current_node

                heapq.heappush(open_list, neighbor_node)

    return None  # No path found