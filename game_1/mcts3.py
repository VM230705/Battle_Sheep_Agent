import STcpClient
import numpy as np
import random
import time

class MCTSNode:
    def __init__(self, playerID, mapStat, sheepStat, movement=None, parent=None):
        self.playerID = playerID
        self.mapStat = mapStat
        self.sheepStat = sheepStat
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.movement = movement  # Store the movement in the node

    def __str__(self):
        return f"PlayerID: {self.playerID}, Visits: {self.visits}, Wins: {self.wins}, Map:\n {self.mapStat}\n, Sheep:\n {self.sheepStat}"
    
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

    def get_available_directions(self, x, y, mapStat):
        directions = []
        direction_mapping = {
            1: (-1, -1), 2: (0, -1), 3: (1, -1),
            4: (-1, 0), 5: (0, 0), 6: (1, 0),
            7: (-1, 1), 8: (0, 1), 9: (1, 1)
        }
        for move_direction, (dx, dy) in direction_mapping.items():
            if move_direction == 5:
                continue
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(mapStat) and 0 <= new_y < len(mapStat[0]) and mapStat[new_x][new_y] == 0:
                directions.append(move_direction)
        return directions


    def expand(self):
        possible_moves = self.get_possible_moves(self.playerID, self.mapStat, self.sheepStat)
        new_nodes = []
        direction_mapping = {
            1: (-1, -1), 2: (0, -1), 3: (1, -1),
            4: (-1, 0), 5: (0, 0), 6: (1, 0),
            7: (-1, 1), 8: (0, 1), 9: (1, 1)
        }
        unique_moves = list(set(possible_moves))  # 去除重复的移动
        new_nodes = []
        for move in unique_moves:
            map_copy = np.copy(self.mapStat)
            sheep_copy = np.copy(self.sheepStat)
            move_position, _, move_direction = move
            x, y = move_position
            dx, dy = direction_mapping[move_direction]
            
            current_available_directions = len(self.get_available_directions(x, y, map_copy))
            
            temp_x, temp_y = x + dx, y + dy
            while 0 <= temp_x < len(map_copy) and 0 <= temp_y < len(map_copy[0]) and map_copy[temp_x][temp_y] == 0:
                new_x, new_y = temp_x, temp_y
                temp_x, temp_y = new_x + dx, new_y + dy
            
            if (new_x - x) <= 1 and (new_y - y) <= 1: # 表示是兩個位置8-adjecency
                current_available_directions -= 1
            
            map_copy[new_x][new_y] = self.playerID
            available_directions = len(self.get_available_directions(new_x, new_y, map_copy))
            if current_available_directions != 0:
                ratio = available_directions/(current_available_directions + available_directions)
                n_sheep = int(sheep_copy[x][y] * ratio)
            else:
                n_sheep = 1
            if n_sheep == 0:
                n_sheep = 1

            sheep_copy[x][y] -= n_sheep # move sheep to new position
            sheep_copy[new_x][new_y] += n_sheep 

            new_movement = (x, y), n_sheep, move_direction  # Store the movement
            new_node = MCTSNode(self.playerID, map_copy, sheep_copy, new_movement, parent=self)
            self.children.append(new_node)
            new_nodes.append(new_node)
            
        return new_nodes
    
    
    def is_game_over(self):
        empty_positions = sum(1 for row in self.mapStat for cell in row if cell == 0)
        players_with_moves = sum(1 for playerID in range(1, 5) if self.get_possible_moves(playerID, self.mapStat, self.sheepStat))
        return empty_positions == 0 or players_with_moves == 0


    def simulate(self):
        if self.parent == None:
            current_player = self.playerID
        else:
            current_player = (self.playerID % 4) + 1
        current_map = np.copy(self.mapStat)
        current_sheep = np.copy(self.sheepStat)
        direction_mapping = {
            1: (-1, -1), 2: (0, -1), 3: (1, -1),
            4: (-1, 0), 5: (0, 0), 6: (1, 0),
            7: (-1, 1), 8: (0, 1), 9: (1, 1)
        }
        while not self.is_game_over():
            possible_moves = self.get_possible_moves(current_player, current_map, current_sheep)
            if not possible_moves:
                break

            # Convert possible moves to a list of tuples (move_info, count)
            moves_with_count = list(possible_moves.items())
            # Extract move_info and counts
            move_infos, counts = zip(*moves_with_count)

            # Adjust counts to be used as weights for random.choice
            weights = [count ** 2 for count in counts]  # Adjust weights (squared to amplify differences)


            # Choose a move based on the adjusted weights
            chosen_move_info = random.choices(move_infos, weights=weights)[0]

            move_position, n_sheep, move_direction = chosen_move_info
            x, y = move_position
            current_available_directions = len(self.get_available_directions(x, y, current_map))
            
            # Update map and sheepStat based on the chosen move
            dx, dy = direction_mapping[move_direction]
            # Move until hitting an obstacle or another player's cell
            temp_x, temp_y = x + dx, y + dy
            while 0 <= temp_x < len(current_map) and 0 <= temp_y < len(current_map[0]) and current_map[temp_x][temp_y] == 0:
                new_x, new_y = temp_x, temp_y
                temp_x, temp_y = new_x + dx, new_y + dy
            
            
            if (new_x - x) <= 1 and (new_y - y) <= 1: # 表示是兩個位置8-adjecency
                current_available_directions -= 1
            

            current_map[new_x][new_y] = current_player
            
            available_directions = len(self.get_available_directions(new_x, new_y, current_map))
            if current_available_directions != 0:
                ratio = available_directions/(current_available_directions + available_directions)
                n_sheep = int(current_sheep[x][y] * ratio)
            else:
                n_sheep = 1
                
            if n_sheep == 0:
                n_sheep = 1
            current_sheep[x][y] -= n_sheep
            current_sheep[new_x][new_y] += n_sheep 
            
            # Switch to the next player
            current_player = (current_player % 4) + 1
        # print("out of while game over", self.is_game_over())
        
        player_scores = {}
        for playerID in range(1, 5):
            player_scores[playerID] = 0
            for region_size in self.get_player_regions(playerID, current_map):
                if region_size > 1:
                    player_scores[playerID] += region_size ** 1.25
                else:
                    player_scores[playerID] += 1
        
        # Determine the winner
        winner = max(player_scores, key=player_scores.get)
        return winner

    
    def calculate_score(self, playerID, mapStat):
        player_score = 0
        for region_size in self.get_player_regions(playerID, mapStat):
            if region_size > 1:
                player_score += region_size ** 1.25
            else:
                player_score += 1
                
        return player_score
    
    def get_player_regions(self, playerID, mapStat):
        # 获取指定玩家的连通区域大小
        regions = []
        visited = set()
        rows, cols = len(mapStat), len(mapStat[0])
        for x in range(rows):
            for y in range(cols):
                if mapStat[x][y] == playerID and (x, y) not in visited:
                    region_size = self.explore_region(x, y, playerID, visited)
                    regions.append(region_size)
        return regions

    def explore_region(self, x, y, playerID, visited):
        # 使用深度优先搜索来探索指定玩家的连通区域大小
        stack = [(x, y)]
        region_size = 0
        rows, cols = len(mapStat), len(mapStat[0])
        while stack:
            curr_x, curr_y = stack.pop()
            if (curr_x, curr_y) not in visited and mapStat[curr_x][curr_y] == playerID:
                visited.add((curr_x, curr_y))
                region_size += 1
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_x, new_y = curr_x + dx, curr_y + dy
                    if 0 <= new_x < rows and 0 <= new_y < cols:
                        stack.append((new_x, new_y))
        return region_size
    

    def backpropagate(self, winner):
        node = self
        while node:
            node.visits += 1
            if node.playerID == winner:
                node.wins += 1
            node = node.parent

    
    def get_possible_moves(self, playerID, mapStat, sheepStat):
        possible_moves = {}
        direction_mapping = {
            1: (-1, -1), 2: (0, -1), 3: (1, -1),
            4: (-1, 0), 5: (0, 0), 6: (1, 0),
            7: (-1, 1), 8: (0, 1), 9: (1, 1)
        }
        rows, cols = len(mapStat), len(mapStat[0])
        
        for x in range(rows):
            for y in range(cols):
                if mapStat[x][y] == playerID and sheepStat[x][y] > 1:
                    n_sheep = sheepStat[x][y]
                    current_available_directions = len(self.get_available_directions(x, y, mapStat))
                    add_weight = 0
                    if current_available_directions <= 2 and n_sheep > 2:
                        add_weight = 1
                        
                    # get current score
                    current_score = self.calculate_score(playerID, mapStat)
                    for move_direction in range(1, 10):
                        if move_direction == 5:
                            continue
                        dx, dy = direction_mapping[move_direction]
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < rows and 0 <= new_y < cols and mapStat[new_x][new_y] == 0:
                            
                            temp_x, temp_y = x + dx, y + dy
                            while 0 <= temp_x < len(mapStat) and 0 <= temp_y < len(mapStat[0]) and mapStat[temp_x][temp_y] == 0:
                                target_x, target_y = temp_x, temp_y
                                temp_x, temp_y = target_x + dx, target_y + dy
                                
                            #  get score after moving
                            mapCopy = np.copy(mapStat)
                            mapCopy[new_x, new_y] = playerID
                            new_score = self.calculate_score(playerID, mapCopy)
                            score_increase = new_score - current_score
                            available_directions = self.get_available_directions(target_x, target_y, mapStat)
                            n_available_directions = len(available_directions)
                            
                            if n_available_directions == 0:
                                n_sheep_for_direction = 1 
                            elif current_available_directions == 1:   
                                n_sheep_for_direction = n_sheep - 1
                            else:
                                ratio = n_available_directions / (current_available_directions + n_available_directions)
                                n_sheep_for_direction = int(n_sheep * ratio)

                                
                            if n_sheep_for_direction == 0:
                                n_sheep_for_direction = 1
                            move_info = ((x, y), n_sheep_for_direction, move_direction)
                            if move_info in possible_moves:
                                possible_moves[move_info] += 1
                            else:
                                possible_moves[move_info] = 1
                            
                            # 加權重
                            if score_increase > 0:
                                possible_moves[move_info] += abs(score_increase - 1) * 8
                            if add_weight:
                                possible_moves[move_info] += 5 * n_sheep * (1/current_available_directions) * 2
                            # Check if the new position can be reached by enemy in one step
                            for enemy_move_direction in range(1, 10):
                                if enemy_move_direction == 5:
                                    continue
                                dx, dy = direction_mapping[enemy_move_direction]
                                # enemy_new_x, enemy_new_y = target_x + dx, target_y + dy
                                enemy_new_x, enemy_new_y = -1, -1
                                temp_x, temp_y = target_x + dx, target_y + dy
                                while 0 <= temp_x < len(mapStat) and 0 <= temp_y < len(mapStat[0]) and mapStat[temp_x][temp_y] == 0:
                                    enemy_new_x, enemy_new_y = temp_x, temp_y
                                    temp_x, temp_y = enemy_new_x + dx, enemy_new_y + dy
                                    
                                if 0 <= enemy_new_x < rows and 0 <= enemy_new_y < cols and mapStat[enemy_new_x][enemy_new_y] != -1 \
                                    and mapStat[enemy_new_x][enemy_new_y] != playerID and sheepStat[enemy_new_x][enemy_new_y] >= 2:
                                    # game3就把sheep stat拿掉
                                    # If an enemy can reach the new position in one step, prioritize this move
                                    possible_moves[move_info] += 3
        return possible_moves
    
    
    def get_movement(self):
        # Return the movement from the current node
        return self.movement

    
def mcts(playerID, mapStat, sheepStat, round):
    root = MCTSNode(playerID, mapStat, sheepStat)
    print(f"Round{round} Map\n", mapStat)
    print(f"Round{round} Sheep\n", sheepStat)
    # input()
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < 2.5:  # Run MCTS for 2.5 seconds
        node = root
        while node.children:
            node = node.select_child()
        if node.visits > 0:
            new_nodes = node.expand()
            if new_nodes:
                node = random.choice(new_nodes)  # Randomly choose a child node to explore
            else:
                # print("no expand node")
                break  # No new nodes, break out of the loop
        
        result = node.simulate()
        node.backpropagate(result)

        iterations += 1

    print("Iterations:", iterations)
    best_child = max(root.children, key=lambda x: x.visits)
    print("Best child:", best_child)
    best_movement = best_child.get_movement()
    print("Best Movement:", best_movement)

    return best_movement


def chebyshev_distance(point1, point2):
    return max(abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))

def InitPos(mapStat):
    print("initial map")
    print(mapStat)
    max_directions = 0
    best_positions = []
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
                        best_positions = [[i, j]]
                    elif directions_count == max_directions and directions_count <= upper_bound and nonborder != 4:
                        best_positions.append([i, j])
    
    print("Best Positions:", best_positions)
    
    # Check if there are any enemies on the board
    enemy_positions = []
    for i in range(rows):
        for j in range(cols):
            if mapStat[i][j] != 0 and mapStat[i][j] != -1:
                enemy_positions.append([i, j])
    # If there are enemy positions, calculate the average distance from each best position to all enemy positions
    if enemy_positions:
        best_pos_avg_distances = []
        for pos in best_positions:
            total_distance = sum([chebyshev_distance(pos, enemy_pos) for enemy_pos in enemy_positions])
            avg_distance = total_distance / len(enemy_positions)
            best_pos_avg_distances.append(avg_distance)
        print("best_pos_avg_distances", best_pos_avg_distances)
        # Find the position with the maximum average distance
        max_avg_distance_idx = best_pos_avg_distances.index(max(best_pos_avg_distances))
        selected_pos = best_positions[max_avg_distance_idx]
    else:
        # If there are no enemy positions, randomly select one of the best positions
        selected_pos = random.choice(best_positions)
    
    print("Selected Position:", selected_pos)
    return selected_pos
        

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
    
    best_move = mcts(playerID, mapStat, sheepStat, round)
    round+=1
    Step = best_move
    STcpClient.SendStep(id_package, Step)
