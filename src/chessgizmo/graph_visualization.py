from collections import Counter, deque
import numpy as np
import pandas as pd
import seaborn as sns
import chess
import chess.svg
import cairosvg
from importlib import resources
from typing import Optional
import boto3
from io import BytesIO
import aioboto3
from botocore.config import Config
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from plotnine import *
from plotnine.exceptions import PlotnineWarning
import warnings
warnings.filterwarnings('ignore', category=PlotnineWarning)

from .config import GizmoConfig


class ChessStorage:
    def __init__(self, config: Optional[GizmoConfig] = None):
        cfg = config or GizmoConfig.from_env()
        self.endpoint = cfg.b2_endpoint
        self.key_id = cfg.b2_key_id
        self.app_key = cfg.b2_application_key
        self.bucket_name = cfg.b2_bucket_name
        self.region = cfg.b2_region

        # Client for synchronous functions
        self.s3_sync = boto3.client(
            service_name='s3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.app_key
        )

        # Configuration for correct S3v4 signing
        self.session = aioboto3.Session()

    def upload_buffer(self, buffer, filename):
        buffer.seek(0)
        self.s3_sync.upload_fileobj(buffer, self.bucket_name, filename)

    def upload_json(self, data_dict, filename):
        """Serializing a dictionary into a JSON string"""
        json_data = json.dumps(data_dict, indent=4, ensure_ascii=False)

        buffer = BytesIO(json_data.encode('utf-8'))
        buffer.seek(0)

        # to transfer ContentType
        self.s3_sync.put_object(
            Bucket=self.bucket_name,
            Key=filename,
            Body=buffer,
            ContentType='application/json'
        )

    async def download_to_buffer(self, filename, buffer):
        """Downloading a file directly to the BytesIO buffer"""
        async with self.session.client(
                service_name='s3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.app_key,
                region_name=self.region
        ) as s3:
            response = await s3.get_object(Bucket=self.bucket_name, Key=filename)
            async with response['Body'] as stream:
                buffer.write(await stream.read())
            buffer.seek(0)

    async def download_json(self, filename):
        """Downloads a JSON file from B2 and returns it as a dict"""
        async with self.session.client(
                service_name='s3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.app_key,
                region_name=self.region
        ) as s3:
            response = await s3.get_object(Bucket=self.bucket_name, Key=filename)
            async with response['Body'] as stream:
                content = await stream.read()

            style_dict = json.loads(content.decode('utf-8'))
            return style_dict


    async def delete_user_folder(self, username: str):
        """Asynchronous deletion of all user files"""
        async with self.session.client(
                service_name='s3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.app_key,
                region_name=self.region
        ) as s3:
            # Listing of objects
            paginator = s3.get_paginator('list_objects_v2')
            async for result in paginator.paginate(Bucket=self.bucket_name, Prefix=f"{username}/"):
                if 'Contents' in result:
                    delete_keys = [{'Key': obj['Key']} for obj in result['Contents']]
                    await s3.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': delete_keys}
                    )
                    print(f'The user folder {username} was successfully deleted')
                else:
                    print(f'The user folder {username} was not found')

    async def get_url(self, filename, expires=3600):
        """Generates a temporary link"""
        async with self.session.client(
                service_name='s3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.app_key,
                region_name=self.region,
                config=Config(signature_version='s3v4')
        ) as s3:
            return await s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': filename},
                ExpiresIn=expires
            )


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
        while self.input_array.shape[1] > self.column_idx and self.column_idx < self.layer:
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
        """Adjusts the colors for each even layer, making them slightly darker."""
        for i in range(len(self.cmap_array)):
            if i % 2 == 1:
                self.cmap_array[i] = self.cmap_array[i] * 0.8  # Reduce brightness by 20%
                self.cmap_array[i][:, 3] = 1.0  # Preserve alpha channel (transparency)


class OpeningTree(PieChart):
    def __init__(self, input_array:np.array, threshold:int, layer:int, user_name:str, turn:str, storage:ChessStorage):
        super().__init__(input_array, threshold, layer)
        self.user_name = user_name
        self.turn = turn
        self.storage = storage
        self.popular_opening = [row[0] for row in self.square_names if len(row) > 0]
        self.background = '#111111'
        self.visualize()

    def draw_chess_board(self, board:chess.Board):
        style = {
            'size': 200,
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
        """Visualizes the diagram and the chessboard."""
        board = chess.Board()
        for ply in self.popular_opening:
            if ply != '~':
                board.push_san(ply)

        fig, ax = plt.subplots()
        size = 0.25
        delta = 0.225

        for row in range(5):
            wedges, texts = ax.pie(self.values_array[row], radius=1 + row * delta, colors=self.cmap_array[row],
                                   wedgeprops=dict(width=size, edgecolor=self.background), startangle=90, counterclock=False, normalize=True)

            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1) / 2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                connectionstyle = 'angle,angleA=0,angleB={}'.format(ang)
                #ax.annotate(self.square_names[row][i][2:4], xy=(x, y), xytext=(x * (1 + row * delta - 0.17), y * (1 + row * delta - 0.17)),
                            #horizontalalignment='center', verticalalignment='center')
                ax.annotate(self.square_names[row][i], xy=(x, y),
                            xytext=(x * (1 + row * delta - 0.17), y * (1 + row * delta - 0.17)),
                            horizontalalignment='center', verticalalignment='center')

        # Draw the chessboard and place it in the center of the diagram
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
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor=self.background)
        filename = f'{self.user_name}/PieChart_for_{self.turn}.png'
        self.storage.upload_buffer(buffer, filename)
        plt.close()


class HeatBoard:
    def __init__(self, username: str, squares: pd.Series, storage=ChessStorage, description: str = 'all') -> None:
        self.username = username
        self.squares = squares
        self.storage = storage
        self.description = description
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


        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.0)
        filename = f'{self.username}/Heatmap_{self.description}.png'
        self.storage.upload_buffer(buffer, filename)
        plt.close()


class MarkedRaincloud:
    def __init__(self, pieces_param_sample: pd.DataFrame, av_player_N, av_player_B,
                 av_player_R_Q, main_rating: int, username: str, game_phase: str, storage:ChessStorage, JACE_COLOR=None) -> None:
        self.pieces_param_sample = pieces_param_sample
        self.av_player_N = av_player_N
        self.av_player_B = av_player_B
        self.av_player_R_Q = av_player_R_Q
        self.main_rating = main_rating
        self.username = username
        self.game_phase = game_phase
        self.storage = storage
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
                    aes(x=1+self.delta_tick, y=self.av_player_B, xend=1-self.delta_tick, yend=self.av_player_B),
                    color=next(self.iter_color), linetype='solid', size=2)
                     + annotate('text', x=1.3,
                                y=self.av_player_B + 1 if self.av_player_B > 15 else self.av_player_B - 1, label='You',
                                size=12,
                                color='#2A2F4F', fontstyle='italic')
                     + geom_segment(
                    aes(x=2+self.delta_tick, y=self.av_player_N, xend=2-self.delta_tick, yend=self.av_player_N),
                    color=next(self.iter_color), linetype='solid', size=2)
                     + annotate('text', x=2.3,
                                y=self.av_player_N + 1 if self.av_player_N > 15 else self.av_player_N - 1, label='You',
                                size=12,
                                color='#2A2F4F', fontstyle='italic')
                     + geom_segment(
                    aes(x=3+self.delta_tick, y=self.av_player_R_Q, xend=3-self.delta_tick, yend=self.av_player_R_Q),
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

        buffer = BytesIO()
        raincloud.save(buffer, format='png', verbose=False)
        filename = f'{self.username}/MarkedRaincloud_in_{self.game_phase}.png'
        self.storage.upload_buffer(buffer, filename)

        plt.clf()
        plt.close()


class VersusViolin:
    def __init__(self, username: str, sample: pd.DataFrame, storage:ChessStorage, normalize=True) -> None:
        self.username = username
        self.sample = sample
        self.storage = storage
        self.scheme = pd.DataFrame(columns=['object for comparison', 'value', 'parameter for comparison', 'class'])
        self.JACE_COLOR = ["#E69F00", "#56B4E9", "#C62E2E", "#257180"]
        self.order = ['bishop', 'knight', 'inc', 'dec']
        self.class_order = ['♞ VS ♝ in opening', '♞ VS ♝ in middle-/endgame', '⚔ VS ☗ in opening',
                            '⚔ VS ☗ in middle-/endgame']

        if normalize:
            self.normalize_attack_defence_columns()
        self.populate_scheme()
        self.draw_violin()

    def normalize_attack_defence_columns(self):
        inc_dec_subset = self.sample.loc[:, ['av_opening_mobility_inc', 'av_opening_mobility_dec',
                                             'av_mittelspiel_endgame_mobility_inc', 'av_mittelspiel_endgame_mobility_dec']]
        knight_bishop_subset = self.sample.loc[:, ['av_opening_knight_activ_coeff', 'av_opening_bishop_activ_coeff',
                                                   'av_mittelspiel_endgame_knight_activ_coeff',
                                                   'av_mittelspiel_endgame_bishop_activ_coeff']]
        inc_dec_subset_normalize = (inc_dec_subset - inc_dec_subset.min().min()) / (inc_dec_subset.max().max() - inc_dec_subset.min().min())
        inc_dec_subset_normalize = inc_dec_subset_normalize * (knight_bishop_subset.max().max() - knight_bishop_subset.min().min()) + knight_bishop_subset.min().min()
        self.sample.loc[:, inc_dec_subset.columns] = inc_dec_subset_normalize

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
                                                              ordered=True)
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

        buffer = BytesIO()
        violin.save(buffer, format='png', verbose=False)
        filename = f"{self.username}/VersusViolin.png"
        self.storage.upload_buffer(buffer, filename)
        buffer.close()


class AchievementsReport:
    def __init__(self, username, win_rating, draw_rating, lose_rating, achicode: list, storage:ChessStorage, language='En_en') -> None:
        self.username = username
        self.win_rating = win_rating
        self.draw_rating = draw_rating
        self.lose_rating = lose_rating
        self.storage = storage
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
        if self.language == 'EN_en':
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
        elif self.language == 'RU_ru':
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
        icon_name = self.path_dict[category][index]
        with resources.path('chessgizmo.data.icons.AchievementsReport', icon_name) as path:
            img = mpimg.imread(str(path))
        ax.imshow(img, extent=pos['img_extent'])
        plt.text(pos['title_x'], pos['title_y'], self.title_dict[category][index], horizontalalignment='left',
                 fontsize=8, color='#CAF0F8', style='italic')
        plt.text(pos['desc_x'], pos['desc_y'], self.description_dict[category][index], horizontalalignment='left',
                 fontsize=8, color='white')

    def draw_report(self):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        fig.patch.set_facecolor('#2E3440')  # Background color of the entire plot
        ax.set_facecolor('#2E3440')  # Background color of the axes

        # Axes settings
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add title text
        plt.text(0.03, 0.9, self.title_text_1, horizontalalignment='left', fontsize=12, color='white')
        plt.text(0.03, 0.6, self.title_text_2, horizontalalignment='left', fontsize=12, color='white')

        # Create a list of active ratings
        ratings = []
        if self.win_rating is not None:
            ratings.append(('Win', self.win_rating, '#386641'))
        if self.draw_rating is not None:
            ratings.append(('Draw', self.draw_rating, 'gray'))
        if self.lose_rating is not None:
            ratings.append(('Lose', self.lose_rating, '#8D0801'))

        n = len(ratings)
        total_width = 0.74  # Total width of the area (0.77 - 0.03)
        gap = 0.025  # Fixed gap between elements

        rect_width = (total_width - gap * (n - 1)) / n

        # Starting position (center if only one element)
        start_x = 0.03 + (total_width - (rect_width * n + gap * (n - 1))) / 2

        # Draw rectangles and text
        for i, (label, rating_value, color) in enumerate(ratings):
            x_pos = start_x + i * (rect_width + gap)
            rect = patches.FancyBboxPatch(
                (x_pos, 0.72), rect_width, 0.12,
                boxstyle="round,pad=0.01", edgecolor='none', facecolor=color
            )
            ax.add_patch(rect)
            center_x = x_pos + rect_width / 2
            plt.text(
                center_x, 0.77, f'{label}\n{rating_value}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, weight='bold', color='white'
            )

        # Positions for images and text
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

        # Add images and text by positions
        for pos, category in zip(positions, categories):
            index = next(self.achi_iterator)
            self.draw_element(ax, index, category, pos)


        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.0)
        # buffer.seek(0)
        file_name = f'{self.username}/AchievementsReport.png'
        self.storage.upload_buffer(buffer, file_name)
        plt.close()




