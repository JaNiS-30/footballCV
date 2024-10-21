import numpy as np
import supervision as sv
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
)
from sports.configs.soccer import SoccerPitchConfiguration

import sys

sys.path.append("../")
from draw_pitch_diagram import draw_pitch_voronoi_diagram_2

CONFIG = SoccerPitchConfiguration()

from utils import save_video


class Pitch:
    def __init__(self):
        pass

    def generate_pitch_view_video(
        self, video_frames, tracks, team_ball_control, output_video_path
    ):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            ball_detections = tracks["ball"][frame_num]
            players_detections = tracks["players"][frame_num]
            referees_detections = tracks["referees"][frame_num]

            frame_ball_xy = np.array(
                [
                    [
                        (ball["bbox"][0] + ball["bbox"][2]) / 2,
                        (ball["bbox"][1] + ball["bbox"][3]) / 2,
                    ]
                    for ball in ball_detections.values()
                ]
            )

            team_1_players = [
                player["position_adjusted"]
                for player in players_detections.values()
                if player["team"] == 1
            ]
            team_2_players = [
                player["position_adjusted"]
                for player in players_detections.values()
                if player["team"] == 2
            ]

            frame_players_team_1_xy = np.array(team_1_players)
            frame_players_team_2_xy = np.array(team_2_players)

            frame_referees_xy = np.array(
                [
                    referee["position_adjusted"]
                    for referee in referees_detections.values()
                ]
            )

            FIELD_WIDTH, FIELD_HEIGHT = 2, 2

            frame_players_team_1_xy = Pitch.normalize_coordinates(
                frame_players_team_1_xy, FIELD_WIDTH, FIELD_HEIGHT
            )
            frame_players_team_2_xy = Pitch.normalize_coordinates(
                frame_players_team_2_xy, FIELD_WIDTH, FIELD_HEIGHT
            )
            frame_ball_xy = Pitch.normalize_coordinates(
                frame_ball_xy, FIELD_WIDTH, FIELD_HEIGHT
            )

            annotated_frame = draw_pitch(CONFIG)

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=frame_ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=annotated_frame,
            )

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=frame_players_team_1_xy,
                face_color=sv.Color.from_hex("00BFFF"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame,
            )

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=frame_players_team_2_xy,
                face_color=sv.Color.from_hex("FF1493"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame,
            )

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=frame_referees_xy,
                face_color=sv.Color.from_hex("FFD700"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame,
            )

            output_video_frames.append(annotated_frame)

        save_video(output_video_frames, output_video_path)

    def generate_voronoi_video(self, video_frames, tracks, output_video_path):
        # """
        # Gera o vídeo com a visualização do diagrama de Voronoi.
        # """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            players_detections = tracks["players"][frame_num]

            # Extrair as coordenadas dos jogadores para cada frame
            frame_players_xy = np.array(
                [player["position_adjusted"] for player in players_detections.values()]
            )

            # 2. Visualizar diagrama de Voronoi
            annotated_frame = draw_pitch(CONFIG)
            annotated_frame = draw_pitch_voronoi_diagram(
                config=CONFIG,
                team_1_xy=frame_players_xy[players_detections["team"] == 0],
                team_2_xy=frame_players_xy[players_detections["team"] == 1],
                team_1_color=sv.Color.from_hex("00BFFF"),
                team_2_color=sv.Color.from_hex("FF1493"),
                pitch=annotated_frame,
            )

            output_video_frames.append(annotated_frame)

        save_video(output_video_frames, output_video_path)

    def generate_voronoi_blend_video(self, video_frames, tracks, output_video_path):
        # """
        # Gera o vídeo com o diagrama de Voronoi e blending.
        # """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            players_detections = tracks["players"][frame_num]
            ball_detections = tracks["ball"][frame_num]

            # Extrair as coordenadas de jogadores e bola para cada frame
            frame_ball_xy = np.array(
                [ball["position_adjusted"] for ball in ball_detections.values()]
            )
            frame_players_xy = np.array(
                [player["position_adjusted"] for player in players_detections.values()]
            )

            # 3. Visualizar diagrama de Voronoi com blend
            annotated_frame = draw_pitch(
                config=CONFIG,
                background_color=sv.Color.WHITE,
                line_color=sv.Color.BLACK,
            )
            annotated_frame = draw_pitch_voronoi_diagram_2(
                config=CONFIG,
                team_1_xy=frame_players_xy[players_detections["team"] == 0],
                team_2_xy=frame_players_xy[players_detections["team"] == 1],
                team_1_color=sv.Color.from_hex("00BFFF"),
                team_2_color=sv.Color.from_hex("FF1493"),
                pitch=annotated_frame,
            )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=frame_ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.WHITE,
                radius=8,
                thickness=1,
                pitch=annotated_frame,
            )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=frame_players_xy[players_detections["team"] == 0],
                face_color=sv.Color.from_hex("00BFFF"),
                edge_color=sv.Color.WHITE,
                radius=16,
                thickness=1,
                pitch=annotated_frame,
            )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=frame_players_xy[players_detections["team"] == 1],
                face_color=sv.Color.from_hex("FF1493"),
                edge_color=sv.Color.WHITE,
                radius=16,
                thickness=1,
                pitch=annotated_frame,
            )

            output_video_frames.append(annotated_frame)

        save_video(output_video_frames, output_video_path)

    def normalize_coordinates(positions, field_width, field_height):
        return np.array([[x * field_width, y * field_height] for x, y in positions])
