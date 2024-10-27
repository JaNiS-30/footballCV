import numpy as np
import cv2
import supervision as sv
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
)
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
import sys

sys.path.append("../")
from draw_pitch_diagram import draw_pitch_voronoi_diagram_2

CONFIG = SoccerPitchConfiguration()

from utils import save_video
from PIL import Image


class Pitch:
    def __init__(self, stability_threshold=5):
        self.player_team_dict = {}  # Time atual dos jogadores
        self.team_change_count = {}  # Contador de estabilidade
        self.stability_threshold = stability_threshold  # Frames para confirmar troca

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
            current_info["count"] = 1  # Resetar o contador se o time mudou
            current_info["current_team"] = new_team_id

        if current_info["count"] >= self.stability_threshold:
            return new_team_id  # Confirmar a mudança

        return self.player_team_dict.get(
            player_id, new_team_id
        )  # Manter o time anterior

    def save_debug_frame_with_points(self, frame, points, name, frame_num):
        """
        Salva um frame com pontos anotados para debugar.
        """
        for point in points:
            x, y = int(point[0]), int(point[1])
            frame = cv2.circle(frame, (x, y), radius=8, color=(0, 255, 0), thickness=-1)

        file_name = f"debug_{name}_frame_{frame_num}.jpg"
        cv2.imwrite(file_name, frame)
        print(f"Salvo: {file_name}")

    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def generate_pitch_view_video(
        self, video_frames, tracks, tracks_pitch, team_ball_control, output_video_path
    ):
        output_video_frames = []

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

            # Desenhar a bola
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=annotated_frame,
            )

            # Desenhar jogadores da equipe 1 com cor personalizada
            for pos, color in zip(pitch_team_1_xy, team_1_colors):
                hex_color = self.rgb_to_hex(color)
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=[pos],
                    face_color=sv.Color.from_hex(hex_color),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=annotated_frame,
                )

            # Desenhar jogadores da equipe 2 com cor personalizada
            for pos, color in zip(pitch_team_2_xy, team_2_colors):
                hex_color = self.rgb_to_hex(color)
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=[pos],
                    face_color=sv.Color.from_hex(hex_color),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=annotated_frame,
                )

            # Desenhar árbitros
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_referees_xy,
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

    def normalize_coordinates(self, positions, field_width, field_height):
        return np.array([[x * field_width, y * field_height] for x, y in positions])
