import cv2
import subprocess
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    intermediate_path = output_video_path.replace(".mp4", "_intermediate.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        intermediate_path,
        fourcc,
        25,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()

    command = [
        "ffmpeg", "-i", intermediate_path,
        "-vcodec", "libx264",  
        "-acodec", "aac",      
        "-strict", "experimental",
        "-y",                  
        output_video_path
    ]
    subprocess.run(command, check=True)
    
    os.remove(intermediate_path)
