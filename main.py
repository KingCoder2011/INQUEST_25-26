from flask import Flask, render_template, request
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math


app = Flask(__name__)
app.secret_key = "your_secret_key"


# ---------- Settings for comparison -----------
ANGLE_TOLERANCE = 80.0  # degrees allowed per joint
SIDE_TO_TRACK = "both"
ANGLE_COLUMNS = ["right_elbow", "right_shoulder_torso", "left_elbow", "left_shoulder_torso"]


mp_pose = mp.solutions.pose
LANDMARK = mp_pose.PoseLandmark


# Movements list for UI if needed
movements = [
    {"display_name": "Cover Drive", "url_name": "cover_drive"},
    {"display_name": "Straight Drive", "url_name": "straight_drive"},
    {"display_name": "Pull Shot", "url_name": "pull_shot"},
    {"display_name": "Square Cut", "url_name": "square_cut"},
    {"display_name": "Helicopter Shot", "url_name": "helicopter_shot"},
]


# ----------- Pose angle utilities -----------


def angle_between_points(a, b, c):
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    dot = np.dot(v1, v2) / (n1 * n2)
    dot = np.clip(dot, -1.0, 1.0)
    return math.degrees(math.acos(dot))


def compute_angles_from_landmarks(lms):
    out = {}
    def get(idx):
        p = lms[idx]
        return np.array([p.x, p.y, p.z])
    try:
        if SIDE_TO_TRACK in ["right", "both"]:
            r_sh = get(LANDMARK.RIGHT_SHOULDER)
            r_el = get(LANDMARK.RIGHT_ELBOW)
            r_wr = get(LANDMARK.RIGHT_WRIST)
            r_hp = get(LANDMARK.RIGHT_HIP)
            out["right_elbow"] = angle_between_points(r_sh, r_el, r_wr)
            out["right_shoulder_torso"] = angle_between_points(r_hp, r_sh, r_el)
        if SIDE_TO_TRACK in ["left", "both"]:
            l_sh = get(LANDMARK.LEFT_SHOULDER)
            l_el = get(LANDMARK.LEFT_ELBOW)
            l_wr = get(LANDMARK.LEFT_WRIST)
            l_hp = get(LANDMARK.LEFT_HIP)
            out["left_elbow"] = angle_between_points(l_sh, l_el, l_wr)
            out["left_shoulder_torso"] = angle_between_points(l_hp, l_sh, l_el)
    except:
        for col in ANGLE_COLUMNS:
            out[col] = np.nan
    return out


def video_to_angles(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results_rows = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            timestamp = frame_idx / fps
            if res.pose_landmarks:
                angles = compute_angles_from_landmarks(res.pose_landmarks.landmark)
            else:
                angles = {c: np.nan for c in ANGLE_COLUMNS}
            row = {"time_s": timestamp, "frame": frame_idx}
            row.update(angles)
            results_rows.append(row)
            frame_idx += 1
    cap.release()
    df = pd.DataFrame(results_rows)
    return df


def compare_angles(df_ref, df_user):
    """Compare angles with temporal normalization for consistency"""
    n_ref = len(df_ref)
    n_user = len(df_user)
    
    # Normalize user video to reference length
    if n_user != n_ref:
        user_indices = np.linspace(0, n_user - 1, n_ref).astype(int)
        df_user_normalized = df_user.iloc[user_indices].reset_index(drop=True)
    else:
        df_user_normalized = df_user
    
    scores = []
    for i in range(n_ref):
        f_ref = df_ref.iloc[i]
        f_user = df_user_normalized.iloc[i]
        correct_joints = 0
        total_joints = 0
        
        for col in ANGLE_COLUMNS:
            a_ref = f_ref[col]
            a_user = f_user[col]
            if not np.isnan(a_ref) and not np.isnan(a_user):
                total_joints += 1
                diff = abs(a_ref - a_user)
                if diff <= ANGLE_TOLERANCE:
                    correct_joints += 1
        
        frame_score = (correct_joints / total_joints * 100) if total_joints > 0 else 0
        scores.append(frame_score)
    
    avg_score = np.mean(scores) if scores else 0
    return avg_score



def get_latest_reference_video():
    """Get the most recently uploaded reference video from python_test folder."""
    upload_folder = os.path.join("static", "python_test")
    if not os.path.exists(upload_folder):
        return None
    video_files = [f for f in os.listdir(upload_folder) if f.endswith('.mp4')]
    if not video_files:
        return None
    video_files_with_path = [os.path.join(upload_folder, f) for f in video_files]
    latest_file = max(video_files_with_path, key=os.path.getmtime)
    return latest_file


# ----------- Animation utilities -----------


def process_video_and_create_animation(video_path, output_animation_path):
    """Process video using MediaPipe and create stick-figure animation using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: " + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stick_frames = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                keypoints_2d = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            else:
                keypoints_2d = []
            stick_frames.append(keypoints_2d)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"Total frames processed: {frame_count}")

    if any(len(k) > 0 for k in stick_frames):
        save_stick_animation_opencv(stick_frames, fps, output_animation_path, width, height)
    else:
        raise RuntimeError("No skeleton data detected in video.")


def save_stick_animation_opencv(frames_keypoints, fps, outname, width, height):
    """Create stick figure animation using OpenCV only."""
    indices = {
        "L_SHO": 11, "R_SHO": 12,
        "L_ELB": 13, "R_ELB": 14,
        "L_WRI": 15, "R_WRI": 16,
        "L_HIP": 23, "R_HIP": 24
    }
    pairs = [
        ("R_HIP", "R_SHO"), ("R_SHO", "R_ELB"), ("R_ELB", "R_WRI"),
        ("L_HIP", "L_SHO"), ("L_SHO", "L_ELB"), ("L_ELB", "L_WRI")
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outname, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not create video writer for {outname}")

    print(f"Creating animation with {len(frames_keypoints)} frames...")

    for frame_idx, kp in enumerate(frames_keypoints):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        for (a, b) in pairs:
            ia = indices[a]
            ib = indices[b]
            if ia < len(kp) and ib < len(kp) and kp[ia] != (0, 0) and kp[ib] != (0, 0):
                pt1 = tuple(map(int, kp[ia]))
                pt2 = tuple(map(int, kp[ib]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
        for joint_name in indices:
            idx = indices[joint_name]
            if idx < len(kp) and kp[idx] != (0, 0):
                pt = tuple(map(int, kp[idx]))
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        out.write(frame)
        if (frame_idx + 1) % 30 == 0:
            print(f"Rendered {frame_idx + 1} frames...")

    out.release()
    print(f"Animation saved successfully to {outname}")


# ----------- Flask routes ---------------


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/scan-video", methods=["GET", "POST"])
def scan_video():
    if request.method == "POST":
        if "video_file" not in request.files:
            return render_template("scan_video.html", error="No file selected.")

        file = request.files["video_file"]
        if file.filename == "":
            return render_template("scan_video.html", error="No file selected.")

        if not file.filename.lower().endswith(".mp4"):
            return render_template(
                "scan_video.html",
                error="Only MP4 video files are allowed. Please upload a .mp4 file.",
            )

        upload_folder = os.path.join("static", "python_test")
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        return render_template(
            "scan_video.html",
            success=f"Video '{file.filename}' uploaded successfully!",
            video_filename=file.filename
        )

    return render_template("scan_video.html", video_filename=None)


@app.route("/generate-animation", methods=["POST"])
def generate_animation():
    video_filename = request.form.get("video_filename")

    if not video_filename:
        return render_template("scan_video.html", error="No video file specified.", video_filename=None)

    video_path = os.path.join("static", "python_test", video_filename)
    if not os.path.exists(video_path):
        return render_template("scan_video.html", error=f"Video file not found: {video_filename}", video_filename=None)

    animations_folder = os.path.join("static", "animations")
    os.makedirs(animations_folder, exist_ok=True)

    base_name = os.path.splitext(video_filename)[0]
    animation_filename = f"{base_name}_animation.mp4"
    animation_path = os.path.join(animations_folder, animation_filename)

    try:
        print(f"Starting animation generation for {video_filename}...")
        process_video_and_create_animation(video_path, animation_path)
        print(f"Animation saved to {animation_path}")
        print(f"Animation exists? {os.path.exists(animation_path)}")

        return render_template(
            "scan_video.html",
            success="Animation generated and saved successfully!",
            video_filename=video_filename,
            animation_saved=True
        )
    except Exception as e:
        import traceback
        print("ERROR while generating animation:")
        traceback.print_exc()
        return render_template(
            "scan_video.html",
            error=f"Error generating animation: {str(e)}",
            video_filename=video_filename
        )


@app.route("/compare-video", methods=["GET", "POST"])
def compare_video():
    if request.method == "POST":
        if "user_video" not in request.files:
            return render_template("compare_video.html", error="No file selected.")
        file = request.files["user_video"]
        if file.filename == "":
            return render_template("compare_video.html", error="No file selected.")
        if not file.filename.lower().endswith(".mp4"):
            return render_template("compare_video.html", error="Only MP4 files allowed.")

        reference_path = get_latest_reference_video()
        
        if not reference_path:
            return render_template("compare_video.html", error="No reference video found. Please upload and generate an animation first.")

        print(f"Using reference video: {reference_path}")

        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        user_video_path = os.path.join(upload_folder, file.filename)
        file.save(user_video_path)

        try:
            print(f"Processing reference video: {reference_path}")
            df_ref = video_to_angles(reference_path)
            print(f"Reference video processed: {len(df_ref)} frames")
            
            print("Processing user video...")
            df_user = video_to_angles(user_video_path)
            print(f"User video processed: {len(df_user)} frames")
            
            print("Comparing angles...")
            score = compare_angles(df_ref, df_user)
            score = round(score, 2)
            
            print(f"Final score: {score}%")
            return render_template("compare_video.html", success=f"Your movement accuracy is {score}%", score=score)
        except Exception as e:
            import traceback
            print("ERROR while comparing videos:")
            traceback.print_exc()
            return render_template("compare_video.html", error=f"Error processing video: {str(e)}")

    return render_template("compare_video.html")


@app.route("/compare-movements")
def compare_movements():
    return render_template("compare_movements.html", movements=movements)


@app.route("/how-to-use")
def how_to_use():
    return render_template("howToUse.html")


# PRESET MOVEMENTS - View Normal Video
@app.route("/view/<string:movement_name>")
def view_normal(movement_name: str):
    """View the actual/normal video for a preset movement"""
    display_name = movement_name.replace("_", " ").title()
    video_path = f"videos/{movement_name}.mp4"
    
    return render_template(
        "view_video.html",
        video_path=video_path,
        video_name=display_name,
        movement_name=movement_name
    )


# PRESET MOVEMENTS - View Animated/Stick Figure Video
@app.route("/view/<string:movement_name>_stick")
def view_animation(movement_name: str):
    """View the stick figure animation video for a preset movement"""
    display_name = movement_name.replace("_", " ").title()
    video_path = f"videos/{movement_name}_stick.mp4"
    
    return render_template(
        "view_video.html",
        video_path=video_path,
        video_name=display_name,
        movement_name=movement_name
    )


# PRESET MOVEMENTS - Practice
@app.route("/practice/<string:movement_name>", methods=["GET", "POST"])
def practice_movement(movement_name: str):
    """Handle practice/comparison for preset movements"""
    display_name = movement_name.replace("_", " ").title()
    
    if request.method == "POST":
        if "user_video" not in request.files:
            return render_template("movement_detail.html", movement_name=movement_name, display_name=display_name, error="No file selected.")
        
        file = request.files["user_video"]
        if file.filename == "":
            return render_template("movement_detail.html", movement_name=movement_name, display_name=display_name, error="No file selected.")
        
        if not file.filename.lower().endswith(".mp4"):
            return render_template("movement_detail.html", movement_name=movement_name, display_name=display_name, error="Only MP4 files allowed.")

        # Reference video is the preset movement video (the normal one, not stick)
        reference_path = os.path.join("static", "videos", f"{movement_name}.mp4")
        
        if not os.path.exists(reference_path):
            return render_template("movement_detail.html", movement_name=movement_name, display_name=display_name, error=f"Reference video not found for {display_name}")

        # Save user's practice video
        upload_folder = os.path.join("static", "practice_uploads")
        os.makedirs(upload_folder, exist_ok=True)
        user_video_path = os.path.join(upload_folder, file.filename)
        file.save(user_video_path)

        try:
            print(f"Processing reference video: {reference_path}")
            df_ref = video_to_angles(reference_path)
            print(f"Reference video processed: {len(df_ref)} frames")
            
            print("Processing user practice video...")
            df_user = video_to_angles(user_video_path)
            print(f"User video processed: {len(df_user)} frames")
            
            print("Comparing angles...")
            score = compare_angles(df_ref, df_user)
            score = round(score, 2)
            
            print(f"Final score: {score}%")
            return render_template(
                "movement_detail.html", 
                movement_name=movement_name, 
                display_name=display_name,
                success=f"Your {display_name} accuracy is {score}%",
                score=score
            )
        except Exception as e:
            import traceback
            print("ERROR while comparing videos:")
            traceback.print_exc()
            return render_template(
                "movement_detail.html", 
                movement_name=movement_name, 
                display_name=display_name,
                error=f"Error processing video: {str(e)}"
            )

    return render_template("movement_detail.html", movement_name=movement_name, display_name=display_name)


if __name__ == "__main__":
    app.run(debug=True)
