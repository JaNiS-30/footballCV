from sklearn.cluster import KMeans
import numpy as np
import supervision as sv


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.team_change_count = {}
        self.team_stable_threshold = 8

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=1,
        )
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        top_half_image = image[0 : int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(top_half_image)

        labels = kmeans.labels_

        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_cluster = kmeans.cluster_centers_[player_cluster]

        return player_cluster

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id, previous_players):
        closest_player_id = self.find_closest_player(player_bbox, previous_players)

        if closest_player_id and closest_player_id in self.player_team_dict:
            stable_team_id = self.player_team_dict[closest_player_id]
        else:
            player_color = self.get_player_color(frame, player_bbox)
            predicted_team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
            stable_team_id = self.update_team_with_stability(player_id, predicted_team_id)

        self.player_team_dict[player_id] = stable_team_id

        return stable_team_id

    def resolve_goalkeepers_team_id(
        self, player_id: str, players: dict, goalkeepers: dict
    ) -> int:
        is_goalkeeper = player_id in goalkeepers
        non_goalkeepers = {k: v for k, v in players.items() if k not in goalkeepers}

        players_xy = np.array(
            [player["position_adjusted"] for player in non_goalkeepers.values()]
        )
        players_class_id = np.array(
            [player["team"] for player in non_goalkeepers.values()]
        )

        team_1_centroid = (
            players_xy[players_class_id == 1].mean(axis=0)
            if np.any(players_class_id == 1)
            else np.zeros(2)
        )
        team_2_centroid = (
            players_xy[players_class_id == 2].mean(axis=0)
            if np.any(players_class_id == 2)
            else np.zeros(2)
        )

        if is_goalkeeper:
            player_position = goalkeepers[player_id]["position_adjusted"]
        else:
            player_position = players[player_id]["position_adjusted"]

        dist_1 = np.linalg.norm(player_position - team_1_centroid)
        dist_2 = np.linalg.norm(player_position - team_2_centroid)

        return 1 if dist_1 < dist_2 else 2

    def find_closest_player(self, current_bbox, previous_players):
        """Encontra o jogador mais próximo com base na distância Euclidiana."""
        current_center = np.array([
            (current_bbox[0] + current_bbox[2]) / 2,
            (current_bbox[1] + current_bbox[3]) / 2,
        ])

        min_distance = float("inf")
        closest_player_id = None

        for player_id, player_info in previous_players.items():
            previous_center = np.array([
                (player_info["bbox"][0] + player_info["bbox"][2]) / 2,
                (player_info["bbox"][1] + player_info["bbox"][3]) / 2,
            ])
            distance = np.linalg.norm(current_center - previous_center)

            if distance < min_distance:
                min_distance = distance
                closest_player_id = player_id

        return closest_player_id

    def update_team_with_stability(self, player_id, new_team_id):
        """Aplica estabilidade na mudança de time."""
        if player_id not in self.team_change_count:
            self.team_change_count[player_id] = {"current_team": new_team_id, "count": 0}

        current_info = self.team_change_count[player_id]

        if current_info["current_team"] == new_team_id:
            current_info["count"] += 1
        else:
            current_info["count"] = 1
            current_info["current_team"] = new_team_id

        if current_info["count"] >= self.team_stable_threshold:
            return new_team_id

        return self.player_team_dict.get(player_id, new_team_id)