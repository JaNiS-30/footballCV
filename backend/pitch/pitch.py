import numpy as np
from collections import deque
from typing import List, Union
import supervision as sv
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_paths_on_pitch,
)
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
import sys

sys.path.append("../")
from draw_pitch_diagram import draw_pitch_voronoi_diagram_2

CONFIG = SoccerPitchConfiguration()

from utils import save_video


class Pitch:
    def __init__(self, stability_threshold=5):
        self.player_team_dict = {}
        self.team_change_count = {}
        self.stability_threshold = stability_threshold
        self.path_raw = []
        self.M = []
        self.max_distance_threshold = 500  # Defina o valor adequado aqui

    def update_team_with_stability(self, player_id, new_team_id):
        """Mantém a consistência do time com base em um limite de estabilidade."""
        if player_id not in self.team_change_count:
            self.team_change_count[player_id] = {
                "current_team": new_team_id,
                "count": 0,
            }

        current_info = self.team_change_count[player_id]

        if current_info["current_team"] == new_team_id:
            current_info["count"] += 1
        else:
            current_info["count"] = 1
            current_info["current_team"] = new_team_id

        if current_info["count"] >= self.stability_threshold:
            return new_team_id

        return self.player_team_dict.get(player_id, new_team_id)

    def generate_pitch_view_video(
        self,
        video_frames,
        tracks,
        tracks_pitch,
        output_video_path,
        output_video_path_voronoi_blend,
    ):
        output_video_frames = []
        output_video_frames_voronoi_blend = []

        for frame_num, frame in enumerate(video_frames):
            ball_detections = tracks["ball"][frame_num]
            players_detections = tracks["players"][frame_num]
            referees_detections = tracks["referees"][frame_num]

            pitch_track = tracks_pitch["keypoints"][frame_num]
            filter = pitch_track["confidences"] > 0.5
            frame_reference_points = pitch_track["filtered_points"][filter]
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            frame_reference_points = np.array(frame_reference_points, dtype=np.float32)
            pitch_reference_points = np.array(pitch_reference_points, dtype=np.float32)

            transformer = ViewTransformer(
                source=frame_reference_points, target=pitch_reference_points
            )

            frame_ball_xy = [
                ball.get("position_adjusted")
                for ball in ball_detections.values()
                if ball.get("position_adjusted") is not None
            ]
            frame_ball_xy = np.array(frame_ball_xy).reshape(-1, 2)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            team_1_players = [
                (player.get("position_adjusted"), player.get("team_color"))
                for player in players_detections.values()
                if player.get("position_adjusted") is not None
                and player.get("team") == 1
            ]

            team_2_players = [
                (player.get("position_adjusted"), player.get("team_color"))
                for player in players_detections.values()
                if player.get("position_adjusted") is not None
                and player.get("team") == 2
            ]

            team_1_positions, team_1_colors = (
                zip(*team_1_players) if team_1_players else ([], [])
            )
            team_2_positions, team_2_colors = (
                zip(*team_2_players) if team_2_players else ([], [])
            )

            team_1_positions = np.array(team_1_positions).reshape(-1, 2)
            team_2_positions = np.array(team_2_positions).reshape(-1, 2)

            pitch_team_1_xy = transformer.transform_points(points=team_1_positions)
            pitch_team_2_xy = transformer.transform_points(points=team_2_positions)

            referees_xy = [
                referee.get("position_adjusted")
                for referee in referees_detections.values()
                if referee.get("position_adjusted") is not None
            ]
            referees_xy = np.array(referees_xy).reshape(-1, 2)
            pitch_referees_xy = transformer.transform_points(points=referees_xy)

            annotated_frame = draw_pitch(CONFIG)
            annotated_frame_voronoi_blend = draw_pitch(
                config=CONFIG,
                background_color=sv.Color.WHITE,
                line_color=sv.Color.BLACK,
            )

            for frame in [annotated_frame, annotated_frame_voronoi_blend]:
                frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.WHITE,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=frame,
                )

            for pos, color in zip(pitch_team_1_xy, team_1_colors):
                hex_color = self.rgb_to_hex(color)
                for frame in [annotated_frame, annotated_frame_voronoi_blend]:
                    frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=[pos],
                        face_color=sv.Color.from_hex(hex_color),
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=frame,
                    )

            for pos, color in zip(pitch_team_2_xy, team_2_colors):
                hex_color = self.rgb_to_hex(color)
                for frame in [annotated_frame, annotated_frame_voronoi_blend]:
                    frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=[pos],
                        face_color=sv.Color.from_hex(hex_color),
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=frame,
                    )

            for frame in [annotated_frame, annotated_frame_voronoi_blend]:
                frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_referees_xy,
                    face_color=sv.Color.from_hex("FFD700"),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=frame,
                )

            annotated_frame_voronoi_blend = draw_pitch_voronoi_diagram_2(
                config=CONFIG,
                team_1_xy=pitch_team_1_xy,
                team_2_xy=pitch_team_2_xy,
                team_1_color=sv.Color.from_hex(self.rgb_to_hex(team_1_colors[0])),
                team_2_color=sv.Color.from_hex(self.rgb_to_hex(team_2_colors[0])),
                pitch=annotated_frame_voronoi_blend,
            )

            output_video_frames.append(annotated_frame)
            output_video_frames_voronoi_blend.append(annotated_frame_voronoi_blend)

        save_video(output_video_frames, output_video_path)
        save_video(output_video_frames_voronoi_blend, output_video_path_voronoi_blend)

    def replace_outliers_based_on_distance(self, positions, distance_threshold):
        last_valid_position = None
        cleaned_positions = []

        for position in positions:
            if position is None or len(position) == 0:
                cleaned_positions.append(np.empty((0, 2), dtype=np.float32))
            else:
                if last_valid_position is None:
                    cleaned_positions.append(position)
                    last_valid_position = position
                else:
                    distance = np.linalg.norm(position - last_valid_position)
                    if distance > distance_threshold:
                        cleaned_positions.append(np.empty((0, 2), dtype=np.float32))
                    else:
                        cleaned_positions.append(position)
                        last_valid_position = position

        return cleaned_positions

    def generate_ball_path_video(
        self, video_frames, tracks, tracks_pitch, output_video_path
    ):
        output_video_frames = []

        # Inicializar path_raw para armazenar os caminhos
        self.path_raw = []

        for frame_num, frame in enumerate(video_frames):
            # Verificar se temos dados da bola e do campo para o frame atual
            if frame_num >= len(tracks["ball"]) or frame_num >= len(
                tracks_pitch["keypoints"]
            ):
                continue

            ball_detections = tracks["ball"][frame_num]
            pitch_track = tracks_pitch["keypoints"][frame_num]

            # Verificar se pitch_track possui dados válidos
            if (
                not pitch_track
                or "confidences" not in pitch_track
                or "filtered_points" not in pitch_track
            ):
                continue

            # Filtrar pontos válidos com confiança > 0.5
            filter = pitch_track["confidences"] > 0.5
            frame_reference_points = np.array(pitch_track["filtered_points"])[filter]
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            # Ignorar frames sem pontos de referência válidos
            if frame_reference_points.size == 0 or pitch_reference_points.size == 0:
                continue

            # Configurar o transformer para transformar as coordenadas
            transformer = ViewTransformer(
                source=frame_reference_points, target=pitch_reference_points
            )
            self.M.append(transformer.m)
            transformer.m = np.mean(np.array(self.M), axis=0)

            # Extrair coordenadas ajustadas da bola e transformá-las
            frame_ball_xy = [
                ball.get("position_adjusted")
                for ball_id, ball in ball_detections.items()
                if ball and ball.get("position_adjusted") is not None
            ]

            if frame_ball_xy:
                frame_ball_xy = np.array(frame_ball_xy).reshape(-1, 2)
                pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
            else:
                pitch_ball_xy = np.empty((0, 2), dtype=np.float32)

            self.path_raw.append(pitch_ball_xy)

        # Processar o caminho e remover outliers
        path = [
            np.empty((0, 2), dtype=np.float32) if coordinates.size < 2 else coordinates
            for coordinates in self.path_raw
        ]
        path = self.replace_outliers_based_on_distance(
            path, self.max_distance_threshold
        )

        # Criar o vídeo com o caminho da bola
        for frame_num, frame in enumerate(video_frames):
            annotated_frame = draw_pitch(CONFIG)
            annotated_frame = draw_paths_on_pitch(
                config=CONFIG, paths=[path], color=sv.Color.WHITE, pitch=annotated_frame
            )
            output_video_frames.append(annotated_frame)

        # Salvar o vídeo
        save_video(output_video_frames, output_video_path)

    def normalize_coordinates(self, positions, field_width, field_height):
        return np.array([[x * field_width, y * field_height] for x, y in positions])

    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
