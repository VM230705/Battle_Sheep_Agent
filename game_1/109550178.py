import STcpClient
import numpy as np
import random
    
def InitPos(mapStat):
    # Select a random border cell to place all 16 sheep for player 1
    print("initial map")
    print(mapStat)
    rows, cols = len(mapStat), len(mapStat[0])
    # 檢查上下邊界
    for i in range(cols):
        if mapStat[0][i] == 0:
            return [0, i]  # 上邊界
        elif mapStat[rows - 1][i] == 0:
            return [rows - 1, i]  # 下邊界
    # 檢查左右邊界
    for i in range(rows):
        if mapStat[i][0] == 0:
            return [i, 0]  # 左邊界
        elif mapStat[i][cols - 1] == 0:
            return [i, cols - 1]  # 右邊界
    # 如果找不到合法的初始位置，返回默認值 [0, 0]
    return [0, 0]

# def GetStep(playerID, mapStat, sheepStat):
#     valid_cells = []
#     print("current map")
#     print(mapStat)
#     print("Sheep")
#     print(sheepStat)
#     input()
#     # Find cells with more than one sheep owned by the current player
#     for i in range(len(mapStat)):
#         for j in range(len(mapStat[0])):
#             if mapStat[i][j] == playerID and sheepStat[i][j] > 1:
#                 valid_cells.append((i, j))

#     # If no valid cells found, pass the turn
#     if not valid_cells:
#         return [(0, 0), 0, 1]

#     # Select a random cell with more than one sheep
#     selected_cell = random.choice(valid_cells)

#     # Split the sheep into two non-empty groups
#     sheep_count = sheepStat[selected_cell[0]][selected_cell[1]]
#     split_count = random.randint(1, sheep_count - 1)  # Ensure non-empty groups
#     remaining_count = sheep_count - split_count

#     # Randomly select a direction to move the group of sheep
#     direction = random.randint(1, 9)

#     # Generate the step
#     step = [(selected_cell[0], selected_cell[1]), split_count, direction]
#     return step
    
    
# 定義遊戲規則
MAX_DEPTH = 3  # 最大搜索深度

# 估值函數：此處假設每個玩家的得分是基於他們的羊群數量
def evaluate(mapStat, playerID):
    return sum(sum(cell == playerID for cell in row) for row in mapStat)

# 最小最大搜索與α-β剪枝
def minimax(mapStat, sheepStat, playerID, depth, alpha, beta):
    if depth == 0:
        return evaluate(mapStat, playerID)

    legal_moves = []
    for i in range(len(mapStat)):
        for j in range(len(mapStat[0])):
            if mapStat[i][j] == playerID and sheepStat[i][j] > 1:
                legal_moves.append((i, j))

    if not legal_moves:
        return evaluate(mapStat, playerID)

    if playerID == 1:  # 當前玩家是我們
        value = float('-inf')
        for move in legal_moves:
            x, y = move
            m = sheepStat[x][y] // 2
            new_mapStat = mapStat.copy()
            new_mapStat[x][y] -= m
            value = max(value, minimax(new_mapStat, sheepStat, playerID, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:  # 當前玩家是對手
        value = float('inf')
        for move in legal_moves:
            x, y = move
            m = sheepStat[x][y] // 2
            new_mapStat = mapStat.copy()
            new_mapStat[x][y] -= m
            value = min(value, minimax(new_mapStat, sheepStat, playerID, depth - 1, alpha, beta))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def alpha_beta_pruning(playerID, mapStat, sheepStat):
    best_value = float('-inf')
    best_move = None
    legal_moves = []
    for i in range(len(mapStat)):
        for j in range(len(mapStat[0])):
            if mapStat[i][j] == playerID and sheepStat[i][j] > 1:
                legal_moves.append((i, j))

    for move in legal_moves:
        x, y = move
        m = int(sheepStat[x][y] // 2)
        new_mapStat = mapStat.copy()
        new_mapStat[x][y] -= m
        
        # 获取可移动的相邻空格
        adjacent_cells = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < len(mapStat) and 0 <= new_y < len(mapStat[0]) and mapStat[new_x][new_y] == 0:
                    adjacent_cells.append((new_x, new_y))
        
        # 计算每个相邻空格的得分
        for new_x, new_y in adjacent_cells:
            value = minimax(new_mapStat, sheepStat, playerID, MAX_DEPTH, float('-inf'), float('inf'))
            if value > best_value:
                best_value = value
                # 计算移动方向
                dx = new_x - x
                dy = new_y - y
                direction = 3 * (dy + 1) + (dx + 1)  # 计算移动方向
                best_move = [(x, y), m, direction]
    print("Best move", best_move)
    print(mapStat)
    input()
    return best_move

    
# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
print("INITIAL POS:", init_pos)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while True:
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = alpha_beta_pruning(playerID, mapStat, sheepStat)
    STcpClient.SendStep(id_package, Step)