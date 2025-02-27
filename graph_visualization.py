from collections import deque
import numpy as np
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from plotnine import *
# from functools import reduce
# from chess import SQUARE_NAMES
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import chess
import chess.svg
import cairosvg
from io import BytesIO
import matplotlib.image as mpimg



class PieChartXXX:
    def __init__(self, input_array, threshold, layer) -> None:
        self.input_array = input_array
        self.threshold = threshold
        self.layer = layer
        self.other_value = 0
        self.other_values = deque()
        self.column_idx = 0
        self.colors = ['#67B99A', '#F9844A', '#8187DC', '#FF7096', '#669BBC', '#BC4749',
                       '#80ED99', '#FFD133', '#CE4257', '#48cae4', '#A7C957', '#E6BEAE',
                       '#2196F3', '#5C677D', '#AD2831', '#606C38', '#FF5C8A']
        self.transparent_color = np.array([1.0, 1.0, 1.0, 0.0])

        self.square_names = [[None], ]
        self.values_array = []
        self.repetitions_list = [0]
        self.plys_sequence_array = []
        self.cmap_array = []

        self.condition_list = []

    def quicksort_dict(self, dict_):
        if len(dict_) <= 1:
            return dict_
        pivot = list(dict_.items())[len(dict_) // 2][1]
        left = {}
        middle = {}
        right = {}
        for key, value in dict_.items():
            if value > self.threshold:
                if value > pivot:
                    left[key] = value
                elif value == pivot:
                    middle[key] = value
                elif value < pivot:
                    right[key] = value
            elif value <= self.threshold:
                self.other_value += value

        sorted_left = self.quicksort_dict(left)
        sorted_right = self.quicksort_dict(right)
        sorted_dict = {**sorted_left, **middle, **sorted_right}

        return sorted_dict

    def func(self):
        while self.input_array.shape[0] > self.column_idx and self.column_idx < self.layer:
            square_name_layer = []
            values_layer = []
            for square_idx, square_name in enumerate(self.square_names[-1]):
                if square_name != '~':
                    if self.column_idx == 0:
                        ply_fetch = self.input_array[:, 0]
                        self.square_names.pop()
                    else:
                        ply_fetch = self.get_fetch(self.input_array, square_idx)
                    ply_counter = Counter(ply_fetch)
                    ply_counter = self.quicksort_dict(ply_counter)

                    square_name_layer.extend(list(ply_counter.keys()))
                    values_layer.extend(list(ply_counter.values()))

                    if self.other_value:
                        self.other_values.append(self.other_value)
                        square_name_layer.append('~')
                        values_layer.append(self.other_value)
                        self.other_value = 0

                elif square_name == '~':  # and self.other_values:
                    square_name_layer.append('~')
                    values_layer.append(self.other_values[0])
                    self.other_values.append(self.other_values[0])
                    self.other_values.popleft()

            self.square_names.append(square_name_layer)
            self.values_array.append(values_layer)

            rept = list(map(lambda x: self.get_repetitions_list(x), values_layer))
            print(self.repetitions_list)
            self.insert_new_row(square_name_layer)
            self.cmap_array[-1] = np.array(list(
                map(lambda x, con: self.transparent_color if con == '~' else x, self.cmap_array[-1],
                    square_name_layer)))
            self.repetitions_list = [0]

            self.pred_square_names = deque(square_name_layer)
            self.pred_values = deque(values_layer)

            self.column_idx += 1

    def add_to_condition_list(self, value, idx):
        if self.condition_list:
            self.condition_list[idx - 1] = value
            self.condition_list = self.condition_list[:idx]
        else:
            self.condition_list.append(value)

    def get_fetch(self, array, square_idx):
        mask = None
        condition_list = self.plys_sequence_array[:, square_idx]
        for idx, condition in enumerate(condition_list):
            if idx == 0:
                mask = array[:, idx] == condition
            else:
                mask &= array[:, idx] == condition
        return array[mask][:, len(condition_list)]

    def insert_new_row(self, square_layer):
        if self.column_idx:
            col = []
            pred_col = []
            pred_array = None
            for i, n in enumerate(self.repetitions_list):
                for _ in range(n):
                    col.append(i + 1)
                    pred_col.append(i)
            pred_array = np.insert(self.plys_sequence_array, col, self.plys_sequence_array[:, pred_col], axis=1)
            #  self.pred_cmap = np.insert(self.cmap_array, col, self.cmap_array[:, pred_col], axis=1)
            # self.pred_cmap = self.copy_colors(self.pred_cmap, col, pred_col)
            self.pred_cmap = np.insert(self.pred_cmap, col, self.pred_cmap[pred_col, :], axis=0)
            self.plys_sequence_array = np.vstack([pred_array, square_layer])
            # self.cmap_array = np.vstack([pred_cmap, self.cmap_array[-1]])
            # self.cmap_array = np.concatenate((self.cmap_array, pred_cmap), axis=0)
            self.cmap_array.append(self.pred_cmap)
        else:
            self.plys_sequence_array = np.array([square_layer])
            cmap = ListedColormap(self.colors)
            custom_colors = cmap(list(range(len(square_layer))))
            #custom_colors = cmap(list(range(len(square_layer) - 1)))
            self.pred_cmap = custom_colors
            self.cmap_array = [custom_colors]

    def reduce_row_values(self, pred_row, value, skip_values=0):
        if pred_row[0]:
            pred_row[0] -= value
            # self.new_row_queue_array.insert()
            return skip_values
        else:
            pred_row.popleft()
            skip_values += 1
            return self.reduce_row_values(pred_row, value, skip_values)

    def get_repetitions_list(self, value):
        if self.column_idx:
            self.pred_values[0] -= value
            if self.pred_values[0]:
                self.repetitions_list[-1] += 1
            else:
                self.pred_square_names.popleft()
                self.pred_values.popleft()
                self.repetitions_list.append(0)

    def copy_colors(self, array: np.ndarray, col, pred_col):
        result = np.copy(array)

        for c, p in zip(col, pred_col):
            result = np.insert(result, c, result[:, p], axis=1)

        return result


class PieChart:
    def __init__(self, input_array, threshold, layer) -> None:
        self.input_array = input_array
        self.threshold = threshold
        self.layer = layer
        self.other_value = 0
        self.other_values = deque()
        self.column_idx = 0
        self.colors = ['#67B99A', '#F9844A', '#8187DC', '#FF7096', '#669BBC', '#BC4749',
                       '#80ED99', '#FFD133', '#CE4257', '#48cae4', '#A7C957', '#E6BEAE',
                       '#2196F3', '#5C677D', '#AD2831', '#606C38', '#FF5C8A']
        self.transparent_color = np.array([1.0, 1.0, 1.0, 0.0])

        self.square_names = [[None], ]
        self.values_array = []
        self.repetitions_list = [0]
        self.plys_sequence_array = []
        self.cmap_array = []
        self.condition_list = []

        self.create_pie_chart()
        self.adjust_colors()

    def quicksort_dict(self, dict_):
        if len(dict_) <= 1:
            return dict_
        pivot = list(dict_.items())[len(dict_) // 2][1]
        left = {}
        middle = {}
        right = {}
        for key, value in dict_.items():
            if value > self.threshold:
                if value > pivot:
                    left[key] = value
                elif value == pivot:
                    middle[key] = value
                elif value < pivot:
                    right[key] = value
            elif value <= self.threshold:
                self.other_value += value

        sorted_left = self.quicksort_dict(left)
        sorted_right = self.quicksort_dict(right)
        sorted_dict = {**sorted_left, **middle, **sorted_right}

        return sorted_dict

    def create_pie_chart(self):
        while self.input_array.shape[0] > self.column_idx and self.column_idx < self.layer:
            square_name_layer = []
            values_layer = []
            for square_idx, square_name in enumerate(self.square_names[-1]):
                if square_name != '~':
                    if self.column_idx == 0:
                        ply_fetch = self.input_array[:, 0]
                        self.square_names.pop()
                    else:
                        ply_fetch = self.get_fetch(self.input_array, square_idx)
                    ply_counter = Counter(ply_fetch)
                    ply_counter = self.quicksort_dict(ply_counter)

                    square_name_layer.extend(list(ply_counter.keys()))
                    values_layer.extend(list(ply_counter.values()))

                    if self.other_value:
                        self.other_values.append(self.other_value)
                        square_name_layer.append('~')
                        values_layer.append(self.other_value)
                        self.other_value = 0

                elif square_name == '~':  # and self.other_values:
                    square_name_layer.append('~')
                    values_layer.append(self.other_values[0])
                    self.other_values.append(self.other_values[0])
                    self.other_values.popleft()

            self.square_names.append(square_name_layer)
            self.values_array.append(values_layer)

            self.insert_new_row(square_name_layer)
            self.cmap_array[-1] = np.array(list(
                map(lambda x, con: self.transparent_color if con == '~' else x, self.cmap_array[-1],
                    square_name_layer)))
            self.repetitions_list = [0]

            self.pred_square_names = deque(square_name_layer)
            self.pred_values = deque(values_layer)

            self.column_idx += 1

    def add_to_condition_list(self, value, idx):
        if self.condition_list:
            self.condition_list[idx - 1] = value
            self.condition_list = self.condition_list[:idx]
        else:
            self.condition_list.append(value)

    def get_fetch(self, array, square_idx):
        mask = None
        condition_list = self.plys_sequence_array[:, square_idx]
        for idx, condition in enumerate(condition_list):
            if idx == 0:
                mask = array[:, idx] == condition
            else:
                mask &= array[:, idx] == condition
        return array[mask][:, len(condition_list)]

    def insert_new_row(self, square_layer):
        if self.column_idx:
            col = []
            pred_col = []
            pred_array = None
            for i, n in enumerate(self.repetitions_list):
                for _ in range(n):
                    col.append(i + 1)
                    pred_col.append(i)
            pred_array = np.insert(self.plys_sequence_array, col, self.plys_sequence_array[:, pred_col], axis=1)
            self.pred_cmap = np.insert(self.pred_cmap, col, self.pred_cmap[pred_col, :], axis=0)
            self.plys_sequence_array = np.vstack([pred_array, square_layer])
            self.cmap_array.append(self.pred_cmap)
        else:
            self.plys_sequence_array = np.array([square_layer])
            cmap = ListedColormap(self.colors)
            custom_colors = cmap(list(range(len(square_layer))))
            self.pred_cmap = custom_colors
            self.cmap_array = [custom_colors]

    def reduce_row_values(self, pred_row, value, skip_values=0):
        if pred_row[0]:
            pred_row[0] -= value
            return skip_values
        else:
            pred_row.popleft()
            skip_values += 1
            return self.reduce_row_values(pred_row, value, skip_values)

    def get_repetitions_list(self, value):
        if self.column_idx:
            self.pred_values[0] -= value
            if self.pred_values[0]:
                self.repetitions_list[-1] += 1
            else:
                self.pred_square_names.popleft()
                self.pred_values.popleft()
                self.repetitions_list.append(0)

    def adjust_colors(self):
        """
        Корректирует цвета для каждого четного слоя, делая их чуть темнее.
        """
        for i in range(len(self.cmap_array)):
            if i % 2 == 1:  # Четные слои (индексы 1, 3, 5 и т.д.)
                self.cmap_array[i] = self.cmap_array[i] * 0.8  # Уменьшаем яркость на 20%
                self.cmap_array[i][:, 3] = 1.0  # Сохраняем альфа-канал (прозрачность)


class OpeningTree(PieChart):
    def __init__(self, input_array:np.array, threshold:int, layer:int, user_name:str, turn:str):
        super().__init__(input_array, threshold, layer)
        self.user_name = user_name
        self.turn = turn
        self.popular_opening = [row[0] for row in self.square_names if len(row) > 0]
        self.background = '#111111'
        self.visualize()

    def draw_chess_board(self, board:chess.Board):
        style = {
            'size': 200,  # Размер меньше для встраивания в диаграмму
            'coordinates': False,
            'colors': {
                'square light': '#f0d9b5',
                'square dark': '#b58863',
                'square light lastmove': '#a9a9a9',
                'square dark lastmove': '#696969',
                'square light check': '#aa3333',
                'square dark check': '#aa1111'
            }
        }
        svg = chess.svg.board(board=board, size=style['size'], coordinates=style['coordinates'], colors=style['colors'])
        png = cairosvg.svg2png(bytestring=svg)
        image = mpimg.imread(BytesIO(png), format='png')
        return image

    def visualize(self):
        """
        Визуализирует диаграмму и шахматную доску.
        """
        # Создаем доску и применяем ходы
        board = chess.Board()
        for ply in self.popular_opening:
            board.push_san(ply)

        fig, ax = plt.subplots()
        size = 0.25
        delta = 0.22

        for row in range(5):
            wedges, texts = ax.pie(self.values_array[row], radius=1 + row * delta, colors=self.cmap_array[row],
                                   wedgeprops=dict(width=size, edgecolor=self.background), startangle=90, counterclock=False, normalize=True)

            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1) / 2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                connectionstyle = 'angle,angleA=0,angleB={}'.format(ang)
                ax.annotate(self.square_names[row][i][2:4], xy=(x, y), xytext=(x * (1 + row * delta - 0.17), y * (1 + row * delta - 0.17)),
                            horizontalalignment='center', verticalalignment='center')

        # Рисуем шахматную доску и размещаем в центре диаграммы
        chess_board_image = self.draw_chess_board(board)
        imagebox = OffsetImage(chess_board_image, zoom=0.64)
        ab = AnnotationBbox(imagebox, (0.5, 0.5), frameon=False, boxcoords='axes fraction', pad=0.5)
        ax.add_artist(ab)

        ax.set_title(
            f'Opening Tree for {self.turn}',
            y=1.3,
            fontsize=16,
            fontweight='bold',
            fontfamily='sans-serif',
            color='white',
        )
        ax.set(aspect='equal')
        plt.savefig(f'./user_data/final_images/{self.user_name}/PieChart_for_{self.turn}.png', bbox_inches='tight', facecolor=self.background)


class HeatBoard:
    def __init__(self, username: str, squares: pd.Series) -> None:
        self.username = username
        self.squares = squares
        self.file_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.rank_names = list(range(8, 0, -1))
        self.heat_map_array = np.zeros([8, 8], dtype=int)
        self.cmap = sns.cubehelix_palette(n_colors=6, start=0.7, rot=0.14, gamma=1.0, hue=1.0, light=0.3, dark=1,
                                          reverse=True, as_cmap=True)
        self.dark_square_tracery = f'' \
                                   f'   --\n' \
                                   f'  ----\n' \
                                   f' ------\n' \
                                   f'--------\n' \
                                   f' ------\n' \
                                   f'  ----\n' \
                                   f'   --\n'
        self.populate_heat_map()
        self.draw_board()

    def populate_heat_map(self):
        counts = self.squares.value_counts().reset_index()
        counts.columns = ['square', 'count']
        file_list = list(map(lambda square: square & 7, counts['square']))
        rank_list = list(map(lambda square: square >> 3, counts['square']))
        count_list = counts['count'].to_list()

        for x, y, value in zip(file_list, rank_list, count_list):
            self.heat_map_array[x, y] = value

    def draw_board(self):
        fig = plt.figure(figsize=(10, 8))
        board_heatmap = sns.heatmap(data=self.heat_map_array, annot=False, cmap=self.cmap, fmt="", linewidths=.5,
                                    linecolor='0')

        board_heatmap.set_xticklabels(self.file_names)
        board_heatmap.set_yticklabels(self.rank_names, rotation=0)
        board_heatmap.tick_params(labelsize=15.0, labelbottom=True, labeltop=True, labelleft=True, labelright=True,
                                  bottom=False, top=False, left=False, right=False)

        for _, spine in board_heatmap.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2.5)

        x_data = [0.083, 0.083, 0.85, 0.85, 0.083]
        y_data = [0.055, 0.943, 0.943, 0.055, 0.055]
        fig.add_artist(Line2D(x_data, y_data, linewidth=2.5, color='0'))

        for j in range(4):
            for i in range(4):
                fig.text(0.172 + i * 0.1548, 0.743 - j * 0.192,
                         self.dark_square_tracery,
                         fontsize=25,
                         fontweight='ultralight',
                         linespacing=0.15,
                         rotation=45)
                fig.text(0.095 + i * 0.1548, 0.647 - j * 0.192,
                         self.dark_square_tracery,
                         fontsize=25,
                         fontweight='ultralight',
                         linespacing=0.15,
                         rotation=45)

        plt.savefig(f'./user_data/final_images/{self.username}/Heatmap.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()


class MarkedRaincloud:
    def __init__(self, pieces_param_sample: pd.DataFrame, av_player_N, av_player_B,
                 av_player_R_Q, main_rating: int, username: str, game_phase: str, JACE_COLOR=None) -> None:
        self.pieces_param_sample = pieces_param_sample
        self.av_player_N = av_player_N
        self.av_player_B = av_player_B
        self.av_player_R_Q = av_player_R_Q
        self.main_rating = main_rating
        self.username = username
        self.game_phase = game_phase
        if JACE_COLOR is None:
            self.JACE_COLOR = ['#CD5C08', '#295F98', '#007A87', '#8CE071', '#7B0051',
                               '#00D1C1', '#FFAA91', '#B4A76C', '#9CA299', '#565A5C',
                               '#00A04B', '#E54C20']
        else:
            self.JACE_COLOR = JACE_COLOR

        self.iter_color = iter(self.JACE_COLOR)
        self.delta_tick = 0.4

        self.pieces_param_schema = pd.DataFrame(columns=['parameters', 'value'])
        self.pieces_param_schema['value'] = self.pieces_param_schema['value'].astype(np.int64)
        self.populate_pieces_param_schema()
        self.draw_raincloud()

    def populate_pieces_param_schema(self):
        for col in self.pieces_param_sample.columns:
            one_parameters_sample = self.pieces_param_sample[col].to_frame()
            one_parameters_sample['parameters'] = col
            one_parameters_sample = one_parameters_sample.rename(columns={col: 'value'})
            self.pieces_param_schema = pd.concat([self.pieces_param_schema, one_parameters_sample], ignore_index=True)

    def draw_raincloud(self):
        game_phase_decr = 'mittelspiel & endgame' if self.game_phase == 'mittelspiel_endgame' else self.game_phase
        raincloud = (ggplot(self.pieces_param_schema, aes(x='parameters', y='value', fill='parameters'))
                     + geom_point(aes(color='parameters'), position=position_jitter(width=0.15), size=0.5, alpha=0.5)
                     + geom_boxplot(width=0.25, outlier_shape='', alpha=0.6)
                     + geom_violin(position=position_nudge(x=0), alpha=0.2, adjust=0.5)
                     + geom_segment(
                    aes(x=1 + self.delta_tick, y=self.av_player_B, xend=1 - self.delta_tick, yend=self.av_player_B),
                    color=next(self.iter_color), linetype='solid', size=2)
                     + annotate('text', x=1.3,
                                y=self.av_player_B + 1 if self.av_player_B > 15 else self.av_player_B - 1, label='You',
                                size=12,
                                color='#2A2F4F', fontstyle='italic')
                     + geom_segment(
                    aes(x=2 + self.delta_tick, y=self.av_player_N, xend=2 - self.delta_tick, yend=self.av_player_N),
                    color=next(self.iter_color), linetype='solid', size=2)
                     + annotate('text', x=2.3,
                                y=self.av_player_N + 1 if self.av_player_N > 15 else self.av_player_N - 1, label='You',
                                size=12,
                                color='#2A2F4F', fontstyle='italic')
                     + geom_segment(
                    aes(x=3 + self.delta_tick, y=self.av_player_R_Q, xend=3 - self.delta_tick, yend=self.av_player_R_Q),
                    color=next(self.iter_color), linetype='solid', size=2)
                     + annotate('text', x=3.3,
                                y=self.av_player_R_Q + 1 if self.av_player_R_Q > 15 else self.av_player_R_Q - 1,
                                label='You', size=12,
                                color='#2A2F4F', fontstyle='italic')
                     + coord_flip()
                     + scale_x_discrete(expand=(0, 0))
                     + scale_fill_manual(values=self.JACE_COLOR)
                     + scale_color_manual(values=self.JACE_COLOR)
                     # + guides(fill=guide_legend(title=''), color=guide_legend(title=''))
                     + guides(fill=guide_legend(), color=guide_legend())
                     + labs(x='',
                            title=f'Your activity figures among the distribution\n by rating {self.main_rating} in {game_phase_decr}')
                     + theme_classic()
                     + theme(legend_position='bottom'))

        raincloud.save(f'./user_data/final_images/{self.username}/MarkedRaincloud_in_{self.game_phase}.png')


class VersusViolin:
    def __init__(self, username: str, sample: pd.DataFrame) -> None:
        self.username = username
        self.sample = sample
        self.scheme = pd.DataFrame(columns=['object for comparison', 'value', 'parameter for comparison', 'class'])
        self.JACE_COLOR = ["#E69F00", "#56B4E9", "#C62E2E", "#257180"]
        self.order = ['bishop', 'knight', 'inc', 'dec']
        self.class_order = ['♞ VS ♝ in opening', '♞ VS ♝ in middle-/endgame', '⚔ VS ☗ in opening',
                            '⚔ VS ☗ in middle-/endgame']

        self.populate_scheme()
        self.draw_violin()

    def populate_scheme(self):
        for col in self.sample.columns:
            one_piece_versus_sample = self.sample[col].to_frame()
            one_piece_versus_sample = one_piece_versus_sample.rename(columns={col: 'value'})

            if col == 'av_opening_knight_activ_coeff':
                one_piece_versus_sample['object for comparison'] = 'knight'
                one_piece_versus_sample['parameter for comparison'] = 'activity coefficient'
                one_piece_versus_sample['class'] = '♞ VS ♝ in opening'

            elif col == 'av_mittelspiel_endgame_knight_activ_coeff':
                one_piece_versus_sample['object for comparison'] = 'knight'
                one_piece_versus_sample['parameter for comparison'] = 'activity coefficient'
                one_piece_versus_sample['class'] = '♞ VS ♝ in middle-/endgame'

            elif col == 'av_opening_bishop_activ_coeff':
                one_piece_versus_sample['object for comparison'] = 'bishop'
                one_piece_versus_sample['parameter for comparison'] = 'activity coefficient'
                one_piece_versus_sample['class'] = '♞ VS ♝ in opening'

            elif col == 'av_mittelspiel_endgame_bishop_activ_coeff':
                one_piece_versus_sample['object for comparison'] = 'bishop'
                one_piece_versus_sample['parameter for comparison'] = 'activity coefficient'
                one_piece_versus_sample['class'] = '♞ VS ♝ in middle-/endgame'

            elif col == 'av_opening_mobility_inc':
                one_piece_versus_sample['object for comparison'] = 'inc'
                one_piece_versus_sample['parameter for comparison'] = 'attack vs defense'
                one_piece_versus_sample['class'] = '⚔ VS ☗ in opening'  # '🤺 VS 🛡️ in opening'

            elif col == 'av_mittelspiel_endgame_mobility_inc':
                one_piece_versus_sample['object for comparison'] = 'inc'
                one_piece_versus_sample['parameter for comparison'] = 'attack vs defense'
                one_piece_versus_sample['class'] = '⚔ VS ☗ in middle-/endgame'  # '🤺 VS 🛡️ in middle-/endgame'

            elif col == 'av_opening_mobility_dec':
                one_piece_versus_sample['object for comparison'] = 'dec'
                one_piece_versus_sample['parameter for comparison'] = 'attack vs defense'
                one_piece_versus_sample['class'] = '⚔ VS ☗ in opening'  # '🤺 VS 🛡️ in opening'

            elif col == 'av_mittelspiel_endgame_mobility_dec':
                one_piece_versus_sample['object for comparison'] = 'dec'
                one_piece_versus_sample['parameter for comparison'] = 'attack vs defense'
                one_piece_versus_sample['class'] = '⚔ VS ☗ in middle-/endgame'  # '🤺 VS 🛡️ in middle-/endgame'

            self.scheme = pd.concat([self.scheme, one_piece_versus_sample], ignore_index=True)

        # Setting categorical data with the desired order for 'object for comparison' and 'class'
        self.scheme['object for comparison'] = pd.Categorical(self.scheme['object for comparison'],
                                                              categories=self.order,
                                                              ordered=True)  # xzzx
        self.scheme['class'] = pd.Categorical(self.scheme['class'], categories=self.class_order, ordered=True)

    def draw_violin(self):
        violin = (ggplot(self.scheme, aes(x='class', y='value', fill='object for comparison'))
                  + geom_violin(position=position_nudge(x=0.4), alpha=0.4, width=0.3)
                  + geom_point(aes(color='object for comparison'),
                               position=position_jitterdodge(jitter_width=0.15, dodge_width=0.3), size=0.3, alpha=0.03,
                               show_legend=False)
                  + geom_boxplot(width=0.3, outlier_shape=None, alpha=0.8)
                  + facet_wrap('parameter for comparison', ncol=2, scales="free_x")
                  + labs(x='', y='estimate')
                  + guides(fill=guide_legend(title=''))
                  + scale_fill_manual(values=self.JACE_COLOR, na_value='#5f5f5f')
                  + scale_color_manual(values=self.JACE_COLOR, na_value='#5f5f5f')
                  + theme_classic(base_size=11)
                  + theme(axis_text_x=element_text(angle=0), legend_position='right')
                  + theme(figure_size=(12, 4))
                  )

        violin.save(f'./user_data/final_images/{self.username}/VersusViolin.png')


class AchievementsReport:
    def __init__(self, username, win_rating, draw_rating, lose_rating, achicode: list, language='En_en') -> None:
        self.username = username
        self.win_rating = win_rating
        self.draw_rating = draw_rating
        self.lose_rating = lose_rating
        self.language = language
        self.achi_iterator = iter(achicode)

        self.path_dict = {
            'best_turn': ['on_the_dark_side_of_the_power.png', 'on_the_light_side_of_the_power.png', 'neutrality.png'],
            'best_piece': ['general_of_the_cavalry.png', 'lurking_dragon.png', 'openline.png'],
            'development': ['strong_development.png', 'slow_development.png'],
            'best_side': ['kingside_castling.png', 'queenside_castling.png', 'king_in_center.png'],
            'attack/defence': ['fire_on_the_board.png', 'impregnable_defense.png'],
            'king_safety': ['king_on_the_castle.png', 'its_time_to_run.png'],
            'endgames': ['frequent_endgames.png', 'quick_knockout.png']
        }
        self.title_text_1 = str()
        self.title_text_2 = str()
        self.title_dict = dict()
        self.description_dict = dict()

        self.create_legend()
        self.draw_report()

    def create_legend(self):
        if self.language == 'En_en':
            self.title_text_1 = 'Average predicted rating when you:'
            self.title_text_2 = 'Your achievements:'
            self.title_dict = {
                'best_turn': ['On the dark side of the power', 'On the light side of the power', 'Neutrality'],
                'best_piece': ['General of the cavalry', 'Lurking dragon', 'Open line'],
                'development': ['Strong development', 'Slow development'],
                'best_side': ['Kingside castling', 'Queenside castling', 'King in center'],
                'attack/defence': ['Fire on the board', 'Impregnable defense'],
                'king_safety': ['King on the castle', 'It\'s time to run'],
                'endgames': ['Frequent endgames', 'Quick knockout']
            }
            self.description_dict = {
                'best_turn': ['You have more wins for black.', 'You have more wins for white.',
                              'You have the most draws.'],
                'best_piece': ['The knight is the most useful \npiece on the board.',
                               'The bishop is the most useful \npiece on the board.',
                               'You position your rooks\nand queen very well.'],
                'development': ['You actively develop your \npieces in the opening.',
                                'The development of the piece \nin the opening is a little slow.'],
                'best_side': ['You castle kingside more often.', 'You castle more often on the queenside.',
                              'Your king often stays in the center.'],
                'attack/defence': ['You like to play open positions.', 'You like to play closed positions.'],
                'king_safety': ['Your king is safe. A material advantage \ndoesn\'t matter if there\'s a checkmate.',
                                'Too many enemy pieces are piling \nup near your king.'],
                'endgames': ['You play endgames quite often.', 'You checkmate before the endgame']
            }
        elif self.language == 'Ru_ru':
            self.title_text_1 = 'Средний предсказанный рейтинг, когда ты:'
            self.title_text_2 = 'Твои достижения:'
            self.title_dict = {
                'best_turn': ['На темной стороне силы', 'На светлой стороне силы', 'Нейтралитет'],
                'best_piece': ['Генерал кавалерии', 'Притаившийся дракон', 'Открытая линия'],
                'development': ['Сильное развитие', 'Медленное развитие'],
                'best_side': ['Рокировка на королевском фланге', 'Рокировка на ферзевом фланге', 'Король в центре'],
                'attack/defence': ['Огонь на доске', 'Неприступная оборона'],
                'king_safety': ['Король в замке', 'Пора бежать'],
                'endgames': ['Частые эндшпили', 'Быстрый нокаут']
            }
            self.description_dict = {
                'best_turn': ['Вы чаще побеждаете чёрными.', 'Вы чаще побеждаете белыми.',
                              'У Вас больше всего ничьих.'],
                'best_piece': ['Конь - самая полезная фигура \nна доске.', 'Слон - самая полезная фигура \nна доске.',
                               'Вы хорошо расставляете \nсвоих ладей и ферзя.'],
                'development': ['Вы активно развиваете свои \nфигуры в дебюте.',
                                'Развитие фигур в дебюте \nпроисходит неспешно.'],
                'best_side': ['Вы чаще делаете рокировку на королевском фланге.',
                              'Вы чаще делаете рокировку на ферзевом фланге.', 'Ваш король часто остается в центре.'],
                'attack/defence': ['Вам нравится играть в открытых позициях.',
                                   'Вам нравится играть в закрытых позициях.'],
                'king_safety': ['Ваш король в безопасности. Какой смысл в \nматериальном преимуществе, если есть мат.',
                                'Слишком много вражеских фигур \nскапливаются вокруг вашего короля.'],
                'endgames': ['Вы часто играете эндшпили.', 'Вы ставите мат до эндшпиля.']
            }

    def draw_element(self, ax, index, category, pos):
        img = mpimg.imread('./icons/AchievementsReport/' + self.path_dict[category][index])
        ax.imshow(img, extent=pos['img_extent'])
        plt.text(pos['title_x'], pos['title_y'], self.title_dict[category][index], horizontalalignment='left',
                 fontsize=8, color='#CAF0F8', style='italic')
        plt.text(pos['desc_x'], pos['desc_y'], self.description_dict[category][index], horizontalalignment='left',
                 fontsize=8, color='white')

    def draw_report(self):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        # Изменение цвета фона
        fig.patch.set_facecolor('#2E3440')  # Цвет фона всего графика
        ax.set_facecolor('#2E3440')  # Цвет фона осей

        # Настройка осей
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Добавление текста заголовка
        plt.text(0.03, 0.9, self.title_text_1, horizontalalignment='left', fontsize=12, color='white')
        plt.text(0.03, 0.6, self.title_text_2, horizontalalignment='left', fontsize=12, color='white')

        # Создание прямоугольников со скругленными краями
        rect1 = patches.FancyBboxPatch((0.03, 0.72), 0.23, 0.12, boxstyle="round,pad=0.01", edgecolor='none',
                                       facecolor='#386641')
        rect2 = patches.FancyBboxPatch((0.285, 0.72), 0.23, 0.12, boxstyle="round,pad=0.01", edgecolor='none',
                                       facecolor='gray')
        rect3 = patches.FancyBboxPatch((0.54, 0.72), 0.23, 0.12, boxstyle="round,pad=0.01", edgecolor='none',
                                       facecolor='#8D0801')

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # Добавление текста внутри прямоугольников
        plt.text(0.145, 0.77, f'Win\n{self.win_rating}', horizontalalignment='center', verticalalignment='center',
                 fontsize=12, weight='bold', color='white')  # , fontfamily='serif'
        plt.text(0.4, 0.77, f'Draw\n{self.draw_rating}', horizontalalignment='center', verticalalignment='center',
                 fontsize=12, weight='bold', color='white')
        plt.text(0.655, 0.77, f'Lose\n{self.lose_rating}', horizontalalignment='center', verticalalignment='center',
                 fontsize=12, weight='bold', color='white')

        # Позиции для изображений и текста
        positions = [
            {'img_extent': [0.03, 0.13, 0.45, 0.55], 'title_x': 0.15, 'title_y': 0.51, 'desc_x': 0.15, 'desc_y': 0.47},
            {'img_extent': [0.03, 0.13, 0.3, 0.4], 'title_x': 0.15, 'title_y': 0.36, 'desc_x': 0.15, 'desc_y': 0.285},
            {'img_extent': [0.03, 0.13, 0.15, 0.25], 'title_x': 0.15, 'title_y': 0.21, 'desc_x': 0.15, 'desc_y': 0.135},
            {'img_extent': [0.03, 0.13, 0.0, 0.1], 'title_x': 0.15, 'title_y': 0.06, 'desc_x': 0.15, 'desc_y': 0.02},
            {'img_extent': [0.63, 0.73, 0.45, 0.55], 'title_x': 0.75, 'title_y': 0.51, 'desc_x': 0.75, 'desc_y': 0.47},
            {'img_extent': [0.63, 0.73, 0.3, 0.4], 'title_x': 0.75, 'title_y': 0.36, 'desc_x': 0.75, 'desc_y': 0.285},
            {'img_extent': [0.63, 0.73, 0.15, 0.25], 'title_x': 0.75, 'title_y': 0.21, 'desc_x': 0.75, 'desc_y': 0.17}
        ]

        categories = ['best_turn', 'best_piece', 'development', 'best_side', 'attack/defence', 'king_safety',
                      'endgames']

        # Добавление изображений и текста по позициям
        for pos, category in zip(positions, categories):
            index = next(self.achi_iterator)
            self.draw_element(ax, index, category, pos)

        plt.savefig(f'./user_data/final_images/{self.username}/AchievementsReport.png', bbox_inches='tight',
                    pad_inches=0.0)





