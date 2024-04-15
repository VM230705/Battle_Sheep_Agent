import STcpClient
import numpy as np
import random

class MCTSNode:
    def __init__(self, playerID, mapStat, sheepStat, parent=None):
        self.playerID = playerID
        self.mapStat = mapStat
        self.sheepStat = sheepStat
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    # def select_child(self):
    #     if not self.children:
    #         self.expand()

    #     log_total_visits = np.log(self.visits)

    #     def ucb(child):
    #         exploitation = child.wins / child.visits if child.visits > 0 else 0
    #         exploration = np.sqrt(log_total_visits / child.visits) if child.visits > 0 else float('inf')
    #         return exploitation + 0.7 * exploration

    #     return max(self.children, key=ucb)

    def select_child(self):
        if not self.children:
            self.expand()

        log_total_visits = np.log(self.visits)

        def ucb(child):
            exploitation = child.wins / child.visits if child.visits > 0 else 0
            exploration = np.sqrt(log_total_visits / child.visits) if child.visits > 0 else float('inf')
            return exploitation + 0.7 * exploration

        max_child = None
        max_ucb = float('-inf')
        for child in self.children:
            child_ucb = ucb(child)
            if child_ucb > max_ucb:
                max_ucb = child_ucb
                max_child = child

        return max_child

    # def expand(self):
    #     possible_moves = self.get_possible_moves()

    #     for move in possible_moves:
    #         map_copy = np.copy(self.mapStat)
    #         sheep_copy = np.copy(self.sheepStat)
    #         new_node = MCTSNode(self.playerID, map_copy, sheep_copy, parent=self)
    #         self.children.append(new_node)
    #     print("expand self.children: ", self.children)
    #     return self.children  # 返回擴展後的子節點列表

    # def expand(self):
    #     possible_moves = self.get_possible_moves()

    #     for move in possible_moves:
    #         map_copy = np.copy(self.mapStat)
    #         sheep_copy = np.copy(self.sheepStat)
    #         new_node = MCTSNode(self.playerID, map_copy, sheep_copy, parent=self)
    #         self.children.append(new_node)
    #     return self.children[0] if self.children else None
    
    def expand(self):
        possible_moves = self.get_possible_moves()
        new_nodes = []
        for move in possible_moves:
            map_copy = np.copy(self.mapStat)
            sheep_copy = np.copy(self.sheepStat)
            new_node = MCTSNode(self.playerID, map_copy, sheep_copy, parent=self)
            self.children.append(new_node)
            new_nodes.append(new_node)
        return new_nodes[0] if new_nodes else None

    
    def simulate(self):
        return random.random()  # Simulate a random play

    # def backpropagate(self, result):
    #     self.visits += 1
    #     self.wins += result
    #     if self.parent:
    #         self.parent.backpropagate(result)

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        node = self.parent
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def get_possible_moves(self):
        possible_moves = []
        direction_mapping = {
            1: (-1, -1), 2: (0, -1), 3: (1, -1),
            4: (-1, 0), 5: (0, 0), 6: (1, 0),
            7: (-1, 1), 8: (0, 1), 9: (1, 1)
        }
        for x in range(15):
            for y in range(15):
                if self.mapStat[x][y] == self.playerID and self.sheepStat[x][y] > 1:
                    n_sheep = self.sheepStat[x][y]
                    half = int(n_sheep // 2)
                    for move_direction in range(1, 10):
                        if move_direction == 5:
                            continue
                        dx, dy = direction_mapping[move_direction]
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < 15 and 0 <= new_y < 15 and self.mapStat[new_x][new_y] == 0:
                            possible_moves.append(((x, y), half, move_direction))
                            
        # print("Current Map\n", self.mapStat)
        # print("Current sheep\n", self.sheepStat)
        return possible_moves
    
def mcts(playerID, mapStat, sheepStat, round):
    root = MCTSNode(playerID, mapStat, sheepStat)
    # print(f"Round{round} map\n", mapStat)
    # print(f"Round{round} Sheep\n", sheepStat)
    # input()
    for _ in range(1000):  # Perform 1000 iterations of MCTS
        node = root
        while node.children:
            node = node.select_child()
        if node.visits > 0:
            node = node.expand()
        result = node.simulate()
        node.backpropagate(result)
    best_child = max(root.children, key=lambda x: x.visits)
    return best_child


def InitPos(mapStat):
    print("initial map")
    print(mapStat)
    max_directions = 0
    best_pos = [0, 0]
    rows, cols = len(mapStat), len(mapStat[0])
    upper_bound = 7  # 初始點最多只有7個方向能走
    # 定義方向字典
    direction_mapping = {
        1: (-1, -1), 2: (0, -1), 3: (1, -1),
        4: (-1, 0), 5: (0, 0), 6: (1, 0),
        7: (-1, 1), 8: (0, 1), 9: (1, 1)
    }
    # 檢查邊界上的每個點
    for i in range(rows):
        for j in range(cols):
            if mapStat[i][j] == 0:  # 如果是空的可移動區域
                directions_count = 0
                nonborder = 0
                for move_direction in range(1, 10):
                    if move_direction == 5:
                        continue
                    dx, dy = direction_mapping[move_direction]
                    new_x, new_y = i + dx, j + dy
                    if 0 <= new_x < rows and 0 <= new_y < cols and mapStat[new_x][new_y] == 0:
                        if move_direction == 2 or move_direction == 4 or move_direction == 6 or move_direction == 8:
                            nonborder += 1
                        directions_count += 1
                    
                if directions_count > max_directions and directions_count <= upper_bound and nonborder != 4:
                    max_directions = directions_count
                    best_pos = [i, j]
    
    print(best_pos)
    print(max_directions)
    return best_pos

    

(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

round = 1
# start game
while True:
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    best_node = mcts(playerID, mapStat, sheepStat, round)
    round+=1
    step = best_node.get_possible_moves()[0]
    STcpClient.SendStep(id_package, step)