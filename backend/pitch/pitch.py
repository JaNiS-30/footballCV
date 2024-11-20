import numpy as np
from collections import deque
from typing import List, Union
import supervision as sv
import matplotlib.pyplot as plt
import seaborn as sns
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
        self.M = deque(maxlen=5)
        self.max_distance_threshold = 500

    def update_team_with_stability(self, player_id, new_team_id):
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
            print("aaaa")
            for frame in [annotated_frame, annotated_frame_voronoi_blend]:
                frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.WHITE,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=frame,
                )
            print("bbbb")

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

            print("cccc")
            if len(pitch_team_2_xy) != len(team_2_colors) or len(pitch_team_2_xy) == 0:
                print(
                    f"Erro: Desbalanceamento ou ausência de dados para a equipe 2. "
                    f"Posições: {len(pitch_team_2_xy)}, Cores: {len(team_2_colors)}"
                )
            else:
                for pos, color in zip(pitch_team_2_xy, team_2_colors):
                    if color is None or len(color) == 0:
                        print("Cor inválida encontrada, ignorando.")
                        continue

                    try:
                        if pos is None or not np.isfinite(pos).all():
                            print("Posição inválida encontrada, ignorando.")
                            continue

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
                    except Exception as e:
                        print(f"Erro ao processar posição ou cor: {e}")

            print("dddd")

            for frame in [annotated_frame, annotated_frame_voronoi_blend]:
                frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_referees_xy,
                    face_color=sv.Color.from_hex("FFD700"),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=frame,
                )
            print("eeee")

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

    def replace_outliers_based_on_distance(
        self, positions: List[np.ndarray], distance_threshold: float = None
    ) -> List[np.ndarray]:
        if distance_threshold is None:
            distance_threshold = self.max_distance_threshold
        last_valid_position = None
        cleaned_positions = []

        for position in positions:
            if len(position) == 0:
                cleaned_positions.append(position)
            else:
                if last_valid_position is None:
                    cleaned_positions.append(position)
                    last_valid_position = position
                else:
                    distance = np.linalg.norm(position - last_valid_position)
                    if distance > distance_threshold:
                        cleaned_positions.append(np.array([], dtype=np.float64))
                    else:
                        cleaned_positions.append(position)
                        last_valid_position = position

        return cleaned_positions

    def generate_ball_tracking_video(
        self,
        video_frames,
        tracks,
        tracks_pitch,
        output_video_path,
    ):
        output_video_frames_ball_tracking = []
        ball_path = deque(maxlen=30)

        for frame_num, frame in enumerate(video_frames):
            ball_detections = tracks["ball"][frame_num]
            pitch_track = tracks_pitch["keypoints"][frame_num]

            filter = pitch_track["confidences"] > 0.5
            frame_reference_points = pitch_track["filtered_points"][filter]
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            frame_reference_points = np.array(frame_reference_points, dtype=np.float32)
            pitch_reference_points = np.array(pitch_reference_points, dtype=np.float32)

            transformer = ViewTransformer(
                source=frame_reference_points, target=pitch_reference_points
            )

            self.M.append(transformer.m)
            transformer.m = np.mean(np.array(self.M), axis=0)

            frame_ball_xy = [
                ball.get("position_adjusted")
                for ball in ball_detections.values()
                if ball.get("position_adjusted") is not None
            ]

            if not frame_ball_xy:
                continue

            frame_ball_xy = np.array(frame_ball_xy).reshape(-1, 2)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            pitch_ball_xy = np.squeeze(pitch_ball_xy)

            ball_path.append(pitch_ball_xy)
            cleaned_path = self.replace_outliers_based_on_distance(list(ball_path))

            annotated_frame = draw_pitch(CONFIG)
            annotated_frame = draw_paths_on_pitch(
                config=CONFIG,
                paths=[cleaned_path],
                color=sv.Color.WHITE,
                pitch=annotated_frame,
            )

            if annotated_frame is not None:
                output_video_frames_ball_tracking.append(annotated_frame)

        save_video(output_video_frames_ball_tracking, output_video_path)

    def generate_team_heatmaps(
        self,
        tracks,
        tracks_pitch,
        output_path_team_1,
        output_path_team_2,
    ):
        team_1_positions_2d = []
        team_2_positions_2d = []

        for frame_num, frame_data in enumerate(tracks_pitch["keypoints"]):
            pitch_track = tracks_pitch["keypoints"][frame_num]
            filter = pitch_track["confidences"] > 0.5
            frame_reference_points = pitch_track["filtered_points"][filter]
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            if not len(frame_reference_points) or not len(pitch_reference_points):
                continue

            frame_reference_points = np.array(frame_reference_points, dtype=np.float32)
            pitch_reference_points = np.array(pitch_reference_points, dtype=np.float32)

            transformer = ViewTransformer(
                source=frame_reference_points, target=pitch_reference_points
            )

            for player_id, player_data in tracks["players"][frame_num].items():
                if player_data["position_adjusted"] is not None:
                    position = np.array(player_data["position_adjusted"]).reshape(1, -1)
                    transformed_position = transformer.transform_points(points=position)
                    if player_data["team"] == 1:
                        team_1_positions_2d.append(transformed_position[0])
                    elif player_data["team"] == 2:
                        team_2_positions_2d.append(transformed_position[0])

        self._generate_heatmap(
            team_1_positions_2d, "Heatmap - Equipe 1", output_path_team_1
        )
        self._generate_heatmap(
            team_2_positions_2d, "Heatmap - Equipe 2", output_path_team_2
        )

    def _generate_heatmap(self, positions, title, output_path):
        if not positions:
            print(f"Sem dados para gerar o {title}.")
            return

        x, y = zip(*positions)

        x = np.array(x) * 0.65 
        y = np.array(y) * 1.2 

        plt.figure(figsize=(12, 7))

        field = draw_pitch(CONFIG)
        plt.imshow(
            field, extent=[0, CONFIG.width, 0, CONFIG.length], aspect="auto"
        )

        ax = plt.gca()
        ax.set_xlim(0, CONFIG.width)
        ax.set_ylim(0, CONFIG.length)

        sns.kdeplot(
            x=x,
            y=y,
            cmap="Reds",  
            fill=True,
            alpha=0.7,  
            bw_adjust=1.5,  
            clip=(
                (0, CONFIG.width),
                (0, CONFIG.length),
            ),  
        )

        plt.title(title)
        plt.axis("off") 
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

    def normalize_coordinates(self, positions, field_width, field_height):
        return np.array([[x * field_width, y * field_height] for x, y in positions])

    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
