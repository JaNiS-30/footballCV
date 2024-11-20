from flask import (
    Flask,
    request,
    render_template,
    send_file,
    send_from_directory,
    url_for,
    Response,
)
import os
import re
from main import main as process_video
import logging

app = Flask(__name__, template_folder="../frontend", static_folder="../frontend")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        logging.error("No file part in the request")
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        logging.error("No selected file")
        return "No selected file", 400
    if file:
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, "output_video.mp4")
        logging.info(f"Saving file to {input_path}")
        file.save(input_path)
        logging.info(f"Processing video from {input_path} to {output_path}")
        try:
            stats = process_video(input_path, output_path)

            logging.info(f"Video processed successfully, sending file {output_path}")

            return {
                "video_url": url_for("get_video", filename="output_video.mp4"),
                "team_passes": stats["team_passes"],
                "team_possession": stats["team_possession"],
                "heatmap_team_1": url_for("get_video", filename="team_1_heatmap.png"),
                "heatmap_team_2": url_for("get_video", filename="team_2_heatmap.png"),
            }
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            return "Failed to process video", 500


@app.route("/output_videos/<filename>")
def get_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
