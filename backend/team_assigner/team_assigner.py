from sklearn.cluster import KMeans
import numpy as np
import supervision as sv


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

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

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id

    def resolve_goalkeepers_team_id(players: dict, goalkeepers: dict) -> dict:
        non_goalkeepers = {k: v for k, v in players.items() if k not in goalkeepers}

        players_xy = np.array(
            [player["position_adjusted"] for player in non_goalkeepers.values()]
        )
        players_class_id = np.array(
            [player["team"] for player in non_goalkeepers.values()]
        )

        goalkeepers_xy = np.array(
            [goalkeeper["position_adjusted"] for goalkeeper in goalkeepers.values()]
        )

        team_0_centroid = players_xy[players_class_id == 0].mean(axis=0)
        team_1_centroid = players_xy[players_class_id == 1].mean(axis=0)

        for goalkeeper_id, goalkeeper_xy in zip(goalkeepers.keys(), goalkeepers_xy):
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)

            if dist_0 < dist_1:
                players[goalkeeper_id]["team"] = 0
            else:
                players[goalkeeper_id]["team"] = 1

        return players
