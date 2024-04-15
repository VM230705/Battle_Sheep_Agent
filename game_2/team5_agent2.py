'''
    Team Name: SheepAgent
    Team ID: 5
    Members: 109550206 陳品劭, 109550178 黃昱翰
'''
import STcpClient
import sys
import copy
import time
import random
import numpy as np
from loguru import logger


logger.remove()
logger.add(sys.stderr, level="INFO")
logger.debug("Client started")

direction_mapping = {
    1: (-1, -1), 2: (0, -1), 3: (1, -1),
    4: (-1, 0), 5: (0, 0), 6: (1, 0),
    7: (-1, 1), 8: (0, 1), 9: (1, 1)
}


def get_player_connected_regions(playerID, mapStat):
    regions = []
    visited = set()
    rows, cols = len(mapStat), len(mapStat[0])

    def explore_region(x, y, playerID, visited, mapStat):
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
    
    for x in range(rows):
        for y in range(cols):
            if mapStat[x][y] == playerID and (x, y) not in visited:
                region_size = explore_region(x, y, playerID, visited, mapStat)
                regions.append(region_size)
    return regions


def calculate_score(playerID, mapStat):
    regions = get_player_connected_regions(playerID, mapStat)
    score = 0
    for region in regions:
        score += region ** 1.25
    return score


class GameState:
    def __init__(self, map_stat, sheep_stat, root_player_id, player_ids):
        self.map_stat = map_stat
        self.sheep_stat = sheep_stat
        self.num_players = 4
        self.current_player = root_player_id
        self.last_move = None
        self.player_ids = player_ids
        self.len_x = len(map_stat)
        self.len_y = len(map_stat[0])
    
    def get_target(self, x, y, direction):
        dx, dy = direction_mapping[direction]
        new_x, new_y = x + dx, y + dy
        target_x, target_y = new_x, new_y
        while self.map_stat[new_x][new_y] != 0:
            target_x, target_y = new_x, new_y
            new_x, new_y = new_x + dx, new_y + dy
        
        return target_x, target_y
    
    def get_possible_directions(self, x, y):
        possible_directions = []
        for direction in range(1, 10):
            if direction == 5:
                continue

            dx, dy = direction_mapping[direction]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.len_x and 0 <= new_y < self.len_y and self.map_stat[new_x][new_y] == 0:
                possible_directions.append(direction)
        return possible_directions

    def get_possible_moves(self, player_id):
        possible_moves = []
        for x in range(self.len_x):
            for y in range(self.len_y):
                num_sheep = int(self.sheep_stat[x][y])
                if self.map_stat[x][y] == player_id and num_sheep > 1:
                    possible_directions = self.get_possible_directions(x, y)
                    if not possible_directions:
                        continue

                    for direction in possible_directions:
                        target_x, target_y = self.get_target(x, y, direction)
                        self.map_stat[target_x][target_y] = player_id
                        target_possible_directions = self.get_possible_directions(target_x, target_y)
                        oringin_possible_directions = self.get_possible_directions(x, y)
                        self.map_stat[target_x][target_y] = 0
                        if not target_possible_directions and not oringin_possible_directions:
                            priority = num_sheep - 1
                        else:
                            priority = len(target_possible_directions) / (len(target_possible_directions) + len(oringin_possible_directions))
                        target_num = int(num_sheep * priority)
                        if target_num < 1:
                            target_num = 1
                        if num_sheep - target_num < 1:
                            target_num = num_sheep - 1
                        
                        out = ""
                        """
                        out += f"| target_num: {target_num}"
                        out += f"| target_possible_directions: {target_possible_directions}"
                        out += f"| oringin_possible_directions: {oringin_possible_directions}"
                        out += f"| priority: {priority}"
                        out += f"| num_sheep: {num_sheep}"
                        out += f"| int(num_sheep * priority): {int(num_sheep * priority)}"
                        """
                        possible_moves.append(((x, y), target_num, direction, out))

        return possible_moves
        
    def get_move_weight(self, player_id, move):
        (x, y), target_num, direction, _ = move
        num_sheep = int(self.sheep_stat[x][y])
        target_x, target_y = self.get_target(x, y, direction)
        oringin_score = calculate_score(player_id, self.map_stat)
        self.map_stat[target_x][target_y] = player_id
        self.sheep_stat[x][y] -= target_num
        target_score = calculate_score(player_id, self.map_stat)
        target_possible_directions = self.get_possible_directions(target_x, target_y)
        #oringin_possible_directions = self.get_possible_directions(x, y)
        self.map_stat[target_x][target_y] = 0
        self.sheep_stat[x][y] += target_num
        weight = (target_score - oringin_score - 1) * 8 + 1

        magic = len(self.get_possible_directions(x, y))
        if magic <= 2 and num_sheep > 2:
            weight += 5 * num_sheep * (2 / magic)
        
        for target_direction in target_possible_directions:
            dx, dy = direction_mapping[target_direction]
            next_target_x, next_target_y = self.get_target(target_x, target_y, target_direction)
            new_x, new_y = next_target_x + dx, next_target_y + dy
            if 0 <= new_x < self.len_x and 0 <= new_y < self.len_y \
                and self.map_stat[new_x][new_y] > 0 and self.map_stat[new_x][new_y] != player_id \
                and self.sheep_stat[new_x][new_y] > 1:
                weight += 3

        return weight
        
    def get_move_weights(self, player_id, possible_moves):
        weights = []
        for move in possible_moves:
            weight = self.get_move_weight(player_id, move)
            weights.append(weight)
        return weights

    def is_terminal(self):
        return not any(self.get_possible_moves(player_id) for player_id in self.player_ids)
    
    def get_next_state(self, move):
        new_state = copy.deepcopy(self)
        new_state.current_player = self.player_ids[(self.player_ids.index(self.current_player) + 1) % self.num_players]
        if move is None:
            new_state.last_move = None
            return new_state
        
        (x, y), target_num, direction, _ = move
        target_x, target_y = self.get_target(x, y, direction)
        new_state.sheep_stat[x][y] -= target_num
        assert new_state.sheep_stat[x][y] >= 1
        assert new_state.sheep_stat[target_x][target_y] == 0
        new_state.sheep_stat[target_x][target_y] = target_num
        new_state.last_move = move
        new_state.map_stat[target_x][target_y] = self.map_stat[x][y]
        return new_state

    def get_scores(self):
        return [calculate_score(player_id, self.map_stat) for player_id in self.player_ids]

    def get_winner(self):
        scores = self.get_scores()
        #logger.info(f"scores: {scores}")
        #logger.info(f"player_ids: {self.player_ids}")
        #logger.info(f"np.argmax(scores): {np.argmax(scores)}")
        return self.player_ids[np.argmax(scores)]

    def get_ranking(self):
        scores = self.get_scores()
        return [self.player_ids[i] for i in np.argsort(scores)[::-1]]

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = [0] * state.num_players
    
    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_moves(self.state.current_player))
    
    def is_terminal(self):
        return self.state.is_terminal()
    
    def get_ucb1(self, player):
        if self.visits == 0:
            return np.inf
        player_index = self.state.player_ids.index(player)
        exploitation = self.wins[player_index] / self.visits
        exploration = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def get_ucb2(self, player):
        if self.visits == 0:
            return np.inf
        player_index = self.state.player_ids.index(player)
        exploitation = self.wins[player_index] / self.visits
        exploration = np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploitation + 0.7 * exploration


class MonteCarloTreeSearch:
    def __init__(self, root_state):
        self.root = Node(root_state)
    
    def select(self, node):
        current_node = node
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return self.expand(current_node)
            elif not current_node.children:
                return self.expand(current_node, SKIP=True)
            else:
                current_node = self.get_best_child(current_node)
        return current_node
    
    def expand(self, node, SKIP=False):
        if SKIP:
            move = None
            next_state = node.state.get_next_state(move)
            child = Node(next_state, parent=node)
            node.children.append(child)
            return child
        
        all_possible_moves = node.state.get_possible_moves(node.state.current_player)
        for move in all_possible_moves:
            next_state = node.state.get_next_state(move)
            child = Node(next_state, parent=node)
            node.children.append(child)
        return random.choice(node.children)
    
    def simulate(self, node):
        current_state = copy.deepcopy(node.state)
        while not current_state.is_terminal():
            possible_moves = current_state.get_possible_moves(current_state.current_player)
            if not possible_moves:
                move = None
            else:
                move = random.choices(possible_moves, weights=current_state.get_move_weights(current_state.current_player, possible_moves))[0]
                #move = random.choice(current_state.get_best_moves(current_state.current_player, possible_moves))
            current_state = current_state.get_next_state(move)
        #return current_state.get_winner()
        return current_state.get_scores(), current_state.get_ranking()
    
    def backpropagate(self, node, result):

        def magic_func(x, k, a, c):
            return c / (1 + np.exp(-k * (x-a)))
        
        while node is not None:
            node.visits += 1
            for i in range(len(node.state.player_ids)):
                scores, ranking = result
                #logger.info(f'result: {result}')
                #logger.info(f'result.index(node.state.player_ids[i]): {result.index(node.state.player_ids[i])}')
                node.wins[i] += magic_func(scores[i], k=(1/7), a=76, c=1) * (4 - ranking.index(node.state.player_ids[i]))
            node = node.parent
    
    def get_best_child(self, node):
        return max(node.children, key=lambda child: child.get_ucb1(node.state.current_player))
    
    def search(self, duration):
        now = time.time()
        while time.time() - now < duration:
            selected_node = self.select(self.root)
            simulation_result = self.simulate(selected_node)
            self.backpropagate(selected_node, simulation_result)
        return max(self.root.children, key=lambda child: child.visits).state.last_move


def get_players(mapStat):
    return [1, 2, 3, 4]
    players = []
    for i in range(len(mapStat)):
        for j in range(len(mapStat[0])):
            if mapStat[i][j] > 0 and mapStat[i][j] not in players:
                players.append(mapStat[i][j])

    assert len(players) == 4

    return players


def get_order(playerID, players):
    players = sorted(players)
    for i in range(len(players)):
        if players[i] == playerID:
            return players[i:] + players[:i]
    return players


def mcts(playerID, mapStat, sheepStat):
    if mapStat is None or sheepStat is None:
        raise ValueError("mapStat and sheepStat cannot be None")
    
    players = get_players(mapStat)
    player_order = get_order(playerID, players)
    initial_state = GameState(mapStat, sheepStat, playerID, player_order)
    mcts = MonteCarloTreeSearch(initial_state)
    best_move = mcts.search(duration=2.9)
    #logger.debug(f"best_move: {best_move}")
    return best_move


def print_map(mapStat):
    if mapStat is None or len(mapStat) == 0 or len(mapStat[0]) == 0:
        return ''

    out = ''
    for j in range(len(mapStat[0])):
        for i in range(len(mapStat)):
            col = int(mapStat[i][j])
            if col == -1:
                out += ' X '
            elif col == 0:
                out += ' . '
            else:
                out += f"{col:2d} "
        out += '\n'
    return out


def print_sheep(mapStat, sheepStat):
    if mapStat is None or sheepStat is None or len(mapStat) == 0 or len(sheepStat) == 0 or len(mapStat[0]) == 0 or len(sheepStat[0]) == 0:
        return ''
    
    out = ''
    for j in range(len(mapStat[0])):
        for i in range(len(mapStat)):
            col = int(mapStat[i][j])
            if col == -1:
                out += ' X '
            elif col == 0:
                out += ' . '
            else:
                out += f"{int(sheepStat[i][j]):2d} "
        out += '\n'
    return out


def find_border(mapStat):
    borders = []
    rows, cols = len(mapStat), len(mapStat[0])
    border_direction_mapping = {
        1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (0, 1)
    }
    for i in range(rows):
        for j in range(cols):
            if mapStat[i][j] == 0:
                for move_direction in range(1, 5):
                    dx, dy = border_direction_mapping[move_direction]
                    new_x, new_y = i + dx, j + dy

                    if 0 <= new_x < rows and 0 <= new_y < cols:
                        if mapStat[new_x][new_y] == -1:
                            borders.append((i, j))
                            break
                    else:
                        borders.append((i, j))
                        break
    return borders


def count_directions(mapStat, borders):
    num_directions = []
    for border in borders:
        x, y = border
        directions_count = 0
        for move_direction in range(1, 10):
            if move_direction == 5:
                continue

            dx, dy = direction_mapping[move_direction]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(mapStat) and 0 <= new_y < len(mapStat[0]) and mapStat[new_x][new_y] == 0:
                directions_count += 1
        num_directions.append(directions_count)
    return num_directions
        

def select_nodes(borders, num_directions, target_nums = [6, 7]):
    target_nodes = []
    for i in range(len(borders)):
        if num_directions[i] in target_nums:
            target_nodes.append(borders[i])
    if not target_nodes:
        return select_nodes(borders, target_nums=[target_nums[0] - 1])
    
    return target_nodes


def get_other_players_position(mapStat):
    other_players_position = []
    for i in range(len(mapStat)):
        for j in range(len(mapStat[0])):
            if mapStat[i][j] in [1, 2, 3, 4]:
                other_players_position.append((i, j))
    return other_players_position


def get_farthest_player_position(nodes, other_players_position):
    distances = []

    def get_distance(x1, y1, x2, y2):
        return max(abs(x2 - x1), abs(y2 - y1))
        #return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    for node in nodes:
        distance = 0.0
        for other_player_position in other_players_position:
            distance += get_distance(node[0], node[1], other_player_position[0], other_player_position[1])
        distances.append(distance)

    return nodes[distances.index(max(distances))]


'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置

'''
def InitPos(mapStat):
    logger.debug("InitPos")
    try:
        borders = find_border(mapStat)
        num_directions = count_directions(mapStat, borders)
        target_nodes = select_nodes(borders, num_directions)
        other_players_position = get_other_players_position(mapStat)
        if not other_players_position:
            init_pos = random.choice(target_nodes)
        else:
            init_pos = get_farthest_player_position(target_nodes, other_players_position)
        return init_pos
    except Exception as e:
        logger.error(f"InitPos error: {e}")
        return None


'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣,
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
'''
def GetStep(playerID, mapStat, sheepStat):
    logger.debug("GetStep")
    try:
        best_move = mcts(playerID, mapStat, sheepStat)
        return best_move
    except Exception as e:
        logger.error(f"GetStep error: {e}")
        return None
    

# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
logger.debug(f"id_package: {id_package}")
logger.debug(f"playerID: {playerID}")
logger.debug(f"mapStat: \n{print_map(mapStat)}")
init_pos = InitPos(mapStat)
logger.debug(f"init_pos: {init_pos}")
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    #logger.debug(f"end_program: {end_program}")
    #logger.debug(f"id_package: {id_package}")
    #logger.debug(f"mapStat: \n{print_map(mapStat)}")
    #logger.debug(f"sheepStat: \n{print_sheep(mapStat, sheepStat)}")

    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
