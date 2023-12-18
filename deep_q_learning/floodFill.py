def flood_fill(board, start):
    """
    Perform a flood fill algorithm from the starting point on the board.
    Args:
    - board: A 2D array representing the game board (0 for empty, 1 for occupied)
    - start: A tuple (x, y) representing the starting point for the flood fill
    Returns:
    - count: The number of cells filled by the algorithm
    """
    if start[0] < 0 or start[0] >= len(board) or start[1] < 0 or start[1] >= len(board[0]):
        return 0  # Start position is out of bounds
    if board[start[0]][start[1]] == 1:
        return 0  # Start position is occupied

    # Directions (right, up, left, down)
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # Initialize the stack with the starting position
    stack = [start]
    count = 0

    while stack:
        x, y = stack.pop()
        if board[x][y] == 0:
            board[x][y] = 2  # Mark the cell as visited
            count += 1
            # Check all adjacent cells
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(board) and 0 <= ny < len(board[0]) and board[nx][ny] == 0:
                    stack.append((nx, ny))
    return count
