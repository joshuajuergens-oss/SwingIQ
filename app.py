import os
import sys
import base64
import traceback
import uuid
import math
import threading
from collections import defaultdict
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import anthropic
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

MEDIAPIPE_AVAILABLE = False
_PoseLandmarker = None
_BaseOptions = None
_PoseLandmarkerOptions = None
_RunningMode = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as _mp_vision
    from mediapipe.tasks.python.core import base_options as _mp_base

    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
    if os.path.exists(_MODEL_PATH):
        _PoseLandmarker = _mp_vision.PoseLandmarker
        _BaseOptions = _mp_base.BaseOptions
        _PoseLandmarkerOptions = _mp_vision.PoseLandmarkerOptions
        _RunningMode = _mp_vision.RunningMode
        MEDIAPIPE_AVAILABLE = True
        print("[startup] MediaPipe pose model loaded ✓")
    else:
        print("[startup] MediaPipe installed but pose_landmarker.task model file not found — skipping pose overlay")
except Exception as _e:
    print(f"[startup] MediaPipe not available ({_e}) — running without pose overlay")

# Load .env from the same folder as this script, regardless of working directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

_key = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"[startup] ANTHROPIC_API_KEY loaded: {repr(_key[:20])}... (len={len(_key)})")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Usage tracking  (1 free analysis per IP per calendar day)
# ---------------------------------------------------------------------------
FREE_LIMIT = 1
_usage_lock = threading.Lock()
# { ip: { "2026-06-15": count } }
_usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


def _get_ip() -> str:
    """Return the real client IP, respecting reverse-proxy headers."""
    return (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.remote_addr
        or "unknown"
    )


def free_analyses_remaining(ip: str) -> int:
    today = str(date.today())
    with _usage_lock:
        used = _usage[ip][today]
    return max(0, FREE_LIMIT - used)


def consume_free_analysis(ip: str) -> bool:
    """Reserve one free analysis slot. Returns True if allowed, False if limit reached."""
    today = str(date.today())
    with _usage_lock:
        if _usage[ip][today] >= FREE_LIMIT:
            return False
        _usage[ip][today] += 1
        return True


def refund_free_analysis(ip: str) -> None:
    """Return a previously reserved free analysis slot (called on error)."""
    today = str(date.today())
    with _usage_lock:
        if _usage[ip][today] > 0:
            _usage[ip][today] -= 1


# ---------------------------------------------------------------------------
# Client factory — uses user's key if provided, else the server key
# ---------------------------------------------------------------------------
_TIMEOUT = anthropic.Timeout(connect=10.0, read=300.0, write=120.0, pool=10.0)
_SERVER_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def make_client(user_key: str = "") -> anthropic.Anthropic:
    key = user_key.strip() if user_key.strip() else _SERVER_KEY
    return anthropic.Anthropic(api_key=key, max_retries=3, timeout=_TIMEOUT)


def validate_key(key: str) -> str | None:
    """Return None if valid, else an error message string."""
    if not key.strip():
        return "No API key provided."
    try:
        c = anthropic.Anthropic(api_key=key.strip(), max_retries=0,
                                 timeout=anthropic.Timeout(connect=8.0, read=15.0, write=8.0, pool=5.0))
        c.messages.create(model="claude-haiku-4-5", max_tokens=5,
                          messages=[{"role": "user", "content": "hi"}])
        return None
    except anthropic.AuthenticationError:
        return "Invalid API key — please check it and try again."
    except Exception as e:
        return f"Could not validate key: {str(e)[:120]}"


# Always return JSON for unhandled errors so the browser never sees raw HTML
@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large. Maximum upload size is 500 MB."}), 413


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _angle(a, b, c) -> float:
    """Angle at point b formed by a-b-c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _lm(landmarks, idx, w, h):
    """Return (x, y) pixel coords for a landmark index (normalized 0-1 coords)."""
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))


def _lm_world(landmarks, idx):
    """Return (x, y) from world landmarks if available."""
    lm = landmarks[idx]
    return (lm.x, lm.y)


def _pose_angles(landmarks, w, h) -> dict:
    """Calculate key golf-relevant joint angles from MediaPipe landmarks."""
    L = landmarks  # shorthand

    def pt(i): return _lm(L, i, w, h)

    angles = {}
    try:
        # Arms
        angles["left_elbow"]   = _angle(pt(11), pt(13), pt(15))   # L shoulder→elbow→wrist
        angles["right_elbow"]  = _angle(pt(12), pt(14), pt(16))   # R shoulder→elbow→wrist
        # Shoulders tilt (vertical angle of shoulder line)
        ls, rs = pt(11), pt(12)
        angles["shoulder_tilt"] = math.degrees(math.atan2(abs(ls[1]-rs[1]), abs(ls[0]-rs[0])))
        # Hips tilt
        lh, rh = pt(23), pt(24)
        angles["hip_tilt"] = math.degrees(math.atan2(abs(lh[1]-rh[1]), abs(lh[0]-rh[0])))
        # Knees
        angles["left_knee"]  = _angle(pt(23), pt(25), pt(27))
        angles["right_knee"] = _angle(pt(24), pt(26), pt(28))
        # Spine angle: midpoint-shoulders to midpoint-hips, relative to vertical
        mid_s = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
        mid_h = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
        dx = mid_s[0] - mid_h[0]
        dy = mid_s[1] - mid_h[1]
        angles["spine_angle"] = math.degrees(math.atan2(abs(dx), abs(dy)))
        # Lead arm (left for right-handed): shoulder to wrist straight-line angle
        angles["lead_arm_elevation"] = _angle(pt(23), pt(11), pt(15))  # hip→shoulder→wrist
    except Exception:
        pass
    return {k: round(v, 1) for k, v in angles.items()}


STAGE_NAMES = ["Setup", "Takeaway", "Backswing", "Top", "Downswing", "Impact/Finish"]


def detect_swing_start(cap, total: int, sample_every: int = 3) -> int:
    """
    Scan the video and return the frame index where significant motion begins
    (i.e. where the swing starts). Falls back to frame 0 if nothing is detected.

    Strategy: compute per-frame motion score (mean absolute difference between
    consecutive downsampled grayscale frames). Collect scores, then find the first
    frame where motion exceeds a dynamic threshold (mean + 1.5 * std of all scores),
    but only after a short quiet period at the start confirms the golfer was still.
    """
    scores = []
    indices = []
    prev_gray = None

    for i in range(0, total, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        # Downsample heavily — we only need motion signal, not detail
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            scores.append(float(np.mean(diff)))
            indices.append(i)
        prev_gray = gray

    if not scores:
        return 0

    scores_arr = np.array(scores)
    threshold = scores_arr.mean() + 1.5 * scores_arr.std()

    # Find the first frame that exceeds threshold, requiring at least a few
    # quiet frames before it (so we don't trigger on camera shake at the very start)
    quiet_frames_needed = max(2, len(scores) // 10)
    quiet_count = 0
    for idx, (frame_idx, score) in enumerate(zip(indices, scores)):
        if score < threshold:
            quiet_count += 1
        elif quiet_count >= quiet_frames_needed:
            # Motion detected after a quiet period — this is the swing start
            # Step back one sample so we catch the very first frame of motion
            swing_start = indices[max(0, idx - 1)]
            print(f"  [motion] swing start detected at frame {swing_start} "
                  f"(score={score:.2f}, threshold={threshold:.2f})")
            return swing_start

    print(f"  [motion] no clear swing start detected — using frame 0")
    return 0


def extract_frames(video_path: str, num_frames: int = 6):
    """
    Detect when the swing starts, then extract evenly-spaced frames from that
    point to the end. Runs MediaPipe pose on each frame.

    Returns:
        frames_b64  : list of base64 JPEG strings (annotated if MediaPipe available)
        pose_summary: human-readable string of per-frame angle data (empty if unavailable)
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], ""

    # Find swing start via motion detection
    swing_start = detect_swing_start(cap, total)
    usable_total = total - swing_start
    if usable_total < num_frames:
        swing_start = 0  # video too short after detected start, use whole thing
        usable_total = total

    indices = [swing_start + int(i * (usable_total - 1) / (num_frames - 1))
               for i in range(num_frames)]
    frames_b64 = []
    all_angles = []   # list of dicts, one per frame

    # Build pose landmarker once for all frames
    pose_ctx = None
    if MEDIAPIPE_AVAILABLE:
        try:
            opts = _PoseLandmarkerOptions(
                base_options=_BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=_RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            pose_ctx = _PoseLandmarker.create_from_options(opts)
        except Exception as e:
            print(f"  [pose] Failed to create landmarker: {e}")

    for frame_num, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))
            h, w = frame.shape[:2]

        frame_angles = {}
        if pose_ctx is not None:
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = pose_ctx.detect(mp_image)

                print(f"  [pose] frame {frame_num}: detected={bool(result.pose_landmarks)}")
                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]  # first (only) pose
                    frame_angles = _pose_angles(lms, w, h)

                    # Draw skeleton manually using landmark connections
                    CONNECTIONS = [
                        (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
                        (11,23),(12,24),(23,24),                   # torso
                        (23,25),(25,27),(24,26),(26,28),           # legs
                        (0,11),(0,12),                             # head-shoulders
                    ]
                    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(lms)}
                    for a, b in CONNECTIONS:
                        if a in pts and b in pts:
                            cv2.line(frame, pts[a], pts[b], (0, 255, 80), 2, cv2.LINE_AA)
                    for pt in pts.values():
                        cv2.circle(frame, pt, 4, (0, 220, 255), -1, cv2.LINE_AA)

                    # Overlay angle text
                    y_pos = 28
                    for name, val in frame_angles.items():
                        label = name.replace("_", " ").title()
                        cv2.putText(frame, f"{label}: {val}°",
                                    (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.50, (0, 255, 120), 1, cv2.LINE_AA)
                        y_pos += 20
            except Exception as e:
                print(f"  [pose] frame {frame_num} error: {e}")

        all_angles.append(frame_angles)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        frames_b64.append(base64.standard_b64encode(buf).decode("utf-8"))
        print(f"  frame {len(frames_b64)}/{num_frames}: {len(buf)//1024} KB  angles={frame_angles}")

    cap.release()
    if pose_ctx is not None:
        pose_ctx.close()

    # Build a compact text summary of pose data for Claude
    pose_summary = ""
    if any(all_angles):
        lines = ["POSE MEASUREMENTS (degrees) — from MediaPipe body-tracking model:"]
        labels = STAGE_NAMES[:len(all_angles)]
        for stage, angles in zip(labels, all_angles):
            if not angles:
                lines.append(f"  {stage}: pose not detected")
                continue
            parts = []
            name_map = {
                "left_elbow": "Left elbow bend",
                "right_elbow": "Right elbow bend",
                "shoulder_tilt": "Shoulder tilt",
                "hip_tilt": "Hip tilt",
                "left_knee": "Left knee bend",
                "right_knee": "Right knee bend",
                "spine_angle": "Spine lean from vertical",
                "lead_arm_elevation": "Lead arm elevation",
            }
            for key, friendly in name_map.items():
                if key in angles:
                    parts.append(f"{friendly}={angles[key]}°")
            lines.append(f"  {stage}: {', '.join(parts)}")
        pose_summary = "\n".join(lines)

    return frames_b64, pose_summary


def build_image_blocks(frames_b64: list[str], label: str) -> list[dict]:
    """Turn a list of base64 frames into Claude content blocks with captions."""
    blocks = []
    for i, b64 in enumerate(frames_b64):
        stage = STAGE_NAMES[i] if i < len(STAGE_NAMES) else f"Frame {i+1}"
        blocks.append({"type": "text", "text": f"{label} — {stage} (frame {i+1} of {len(frames_b64)})"})
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
    return blocks


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/usage-status")
def usage_status():
    ip = _get_ip()
    remaining = free_analyses_remaining(ip)
    return jsonify({"free_remaining": remaining, "free_limit": FREE_LIMIT})


@app.route("/debug-status")
def debug_status():
    return jsonify({
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "model_path": _MODEL_PATH if MEDIAPIPE_AVAILABLE else None,
        "model_exists": os.path.exists(_MODEL_PATH) if MEDIAPIPE_AVAILABLE else False,
        "server_key_len": len(os.environ.get("ANTHROPIC_API_KEY", "")),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    front_file = request.files.get("front_video")
    back_file  = request.files.get("back_video")

    if not front_file and not back_file:
        return jsonify({"error": "Please upload at least one video (front or back view)."}), 400

    # ── Key / free-usage check ─────────────────────────────────────────────
    user_key = request.form.get("api_key", "").strip()
    ip = _get_ip()
    using_free = False

    if user_key:
        # Validate the key before doing expensive work
        err = validate_key(user_key)
        if err:
            return jsonify({"error": err}), 401
        ai_client = make_client(user_key)
    else:
        # Try to consume a free analysis
        if not consume_free_analysis(ip):
            return jsonify({
                "error": "free_limit_reached",
                "message": "You've used your 1 free analysis for today. "
                           "Enter your Anthropic API key to continue — it's free to sign up at console.anthropic.com.",
            }), 402
        ai_client = make_client()   # server key
        using_free = True

    club = request.form.get("club", "").strip()

    uid = uuid.uuid4().hex
    saved_paths = {}

    for key, f in [("front", front_file), ("back", back_file)]:
        if f and f.filename:
            ext = os.path.splitext(secure_filename(f.filename))[1] or ".mp4"
            path = os.path.join(UPLOAD_FOLDER, f"{key}_{uid}{ext}")
            f.save(path)
            saved_paths[key] = path

    try:
        front_frames, front_pose = extract_frames(saved_paths["front"], num_frames=6) if "front" in saved_paths else ([], "")
        back_frames,  back_pose  = extract_frames(saved_paths["back"],  num_frames=6) if "back"  in saved_paths else ([], "")
    except Exception as exc:
        traceback.print_exc()
        if using_free:
            refund_free_analysis(ip)
        return jsonify({"error": f"Frame extraction failed: {str(exc)}"}), 422
    finally:
        for p in saved_paths.values():
            try:
                os.remove(p)
            except OSError:
                pass

    if not front_frames and not back_frames:
        if using_free:
            refund_free_analysis(ip)
        return jsonify({"error": "Could not extract frames from the video. Check the file format (MP4/MOV recommended)."}), 422

    pose_data_block = ""
    if front_pose or back_pose:
        parts = []
        if front_pose:
            parts.append("FRONT VIEW:\n" + front_pose)
        if back_pose:
            parts.append("BACK VIEW:\n" + back_pose)
        pose_data_block = "\n\n".join(parts)

    if front_frames and back_frames:
        angle_desc = "one from the **front (face-on)** and one from the **back (down-the-line)**"
        angle_note = "Both angles are provided. Front-view frames appear first, then back-view frames."
    elif front_frames:
        angle_desc = "the **front (face-on)** angle only"
        angle_note = "Only the front (face-on) view is provided. Note any limitations this creates and focus on what is clearly visible from this angle."
    else:
        angle_desc = "the **back (down-the-line)** angle only"
        angle_note = "Only the back (down-the-line) view is provided. Note any limitations this creates and focus on what is clearly visible from this angle."

    # Build club-aware context string
    if club:
        club_context = (
            f"The golfer is hitting a **{club}**. "
            + {
                "Driver":         "For the driver, pay close attention to tee height, ball position (forward in stance), spine tilt away from target at address, wide arc, lag preservation, and full extension through impact. Weight should load fully into the trail side on the backswing.",
                "3-Wood":         "For a 3-wood, note ball position (slightly inside lead heel), sweeping angle of attack, spine tilt, and whether the golfer is trying to 'help' the ball up rather than sweeping through.",
                "5-Wood":         "For a 5-wood, assess ball position, sweep vs. descending blow, and whether the golfer maintains spine angle through impact.",
                "Hybrid":         "For a hybrid, check ball position (middle-forward), slight descending blow, and whether the golfer is making a sweeping or iron-like motion.",
                "3-Iron":         "For a long iron, focus on ball position, maintaining lag, spine angle, and avoiding early extension or casting.",
                "4-Iron":         "For a long iron, focus on ball position, lag retention, and a slight descending blow. Common fault: flipping at impact.",
                "5-Iron":         "For a mid-iron, check ball position (center-forward), descending angle of attack, and lag. Note hip clearance and shaft lean at impact.",
                "6-Iron":         "For a mid-iron, check ball position, shaft lean at impact, and whether the divot is in front of the ball position.",
                "7-Iron":         "For a 7-iron, check ball position (slightly forward of center), shaft lean, divot location, and hip rotation speed.",
                "8-Iron":         "For a short iron, check steeper angle of attack, ball position near center, shaft lean, and control of swing length.",
                "9-Iron":         "For a short iron, note the steeper attack angle, centered ball position, shaft lean at impact, and abbreviated but balanced finish.",
                "Pitching Wedge": "For the pitching wedge, assess shaft lean, ball position (center), angle of attack, and whether the golfer is decelerating into impact.",
                "Gap Wedge":      "For the gap wedge, focus on shaft lean, controlled swing length, angle of attack, and face angle at impact.",
                "Sand Wedge":     "For the sand wedge, note whether this is a full swing or partial shot — check shaft lean, face angle, and whether the bounce is being used correctly.",
                "Lob Wedge":      "For the lob wedge, pay close attention to face angle (open?), swing path, shaft lean (minimal for high shots), and whether the golfer is trying to scoop.",
                "Putter":         "For the putting stroke, focus on: eye position over the ball, shoulder rocking vs. hands/wrists, putter path (straight or slight arc), face angle at impact, tempo and rhythm, and follow-through length relative to backswing.",
            }.get(club, f"Apply club-appropriate expectations for a {club} regarding ball position, angle of attack, and finish.")
        )
    else:
        club_context = "The specific club used was not provided — give general swing analysis applicable to any full-swing club."

    pose_intro = ""
    if pose_data_block:
        pose_intro = (
            "\n\nA body-tracking model (MediaPipe) has measured the following joint angles at each stage of the swing. "
            "Use these numbers to make your feedback more precise — reference specific angles when explaining what the golfer is doing and what ideal looks like.\n\n"
            + pose_data_block
        )

    # Shared context block given to every agent
    shared_intro = (
        f"Below are evenly-spaced frames extracted from a slow-motion video of a golf swing — {angle_desc}. "
        f"\n\n**Club:** {club if club else 'Not specified'}\n"
        f"{club_context}\n\n"
        f"{angle_note}"
        f"{pose_intro}"
    )

    image_blocks = []
    if front_frames:
        image_blocks.extend(build_image_blocks(front_frames, "FRONT VIEW"))
    if back_frames:
        image_blocks.extend(build_image_blocks(back_frames, "BACK VIEW"))

    # --- Multi-agent panel: three specialists look at the same swing ---
    AGENTS = [
        {
            "name": "Body & Posture Coach",
            "persona": (
                "You are a golf biomechanics specialist. Your ONLY focus is the golfer's BODY: "
                "posture, spine angle, balance, weight shift, hip rotation, shoulder turn, knee flex, "
                "and the order in which body parts move (sequencing). Ignore the club and hands except "
                "where they reveal what the body is doing. Lean heavily on the pose measurements if provided."
            ),
        },
        {
            "name": "Club & Hands Coach",
            "persona": (
                "You are a golf club-delivery specialist. Your ONLY focus is the CLUB and HANDS: "
                "grip, wrist hinge, club path, swing plane, clubface angle, lag, casting, shaft lean at impact, "
                "and release through the ball. Ignore body posture except where it affects club delivery."
            ),
        },
        {
            "name": "Tempo & Rhythm Coach",
            "persona": (
                "You are a golf tempo and motion-flow specialist. Your ONLY focus is RHYTHM, TIMING, and FLOW: "
                "the smoothness of transitions between swing stages, backswing-to-downswing ratio, signs of rushing "
                "or hesitation, balance through the finish, and overall athletic fluidity across the frame sequence. "
                "Compare consecutive frames to judge how the motion develops over time."
            ),
        },
    ]

    # Haiku for agents (fast, cheap), Opus for final synthesis (best quality)
    AGENT_MODEL     = "claude-opus-4-8"
    SYNTHESIS_MODEL = "claude-opus-4-8"

    def run_agent(agent: dict) -> str:
        blocks = [{"type": "text", "text": agent["persona"] + "\n\n" + shared_intro}]
        blocks.extend(image_blocks)
        blocks.append({
            "type": "text",
            "text": (
                "Examine the swing strictly from your specialty. List EVERY issue you can find in your area — "
                "do not limit yourself to the most important ones. Minor flaws count too. For each issue give:\n"
                "- A short plain-English title\n"
                "- What you see (1-2 sentences, simple language a non-golfer understands; explain any golf term in brackets)\n"
                "- How serious it is: MAJOR, MODERATE, or MINOR\n\n"
                "Also list anything in your specialty the golfer does WELL.\n"
                "Be honest and thorough — another coach will cross-check your findings."
            ),
        })
        # Plain call — adaptive thinking handled by synthesis step
        response = ai_client.messages.create(
            model=AGENT_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": blocks}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    total_kb = sum(len(b) * 3 // 4 // 1024 for b in front_frames + back_frames)
    print(f"Running 3 specialist agents [{AGENT_MODEL}] in parallel "
          f"({len(front_frames)} front + {len(back_frames)} back frames, ~{total_kb} KB each) "
          f"| free={using_free}")

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(run_agent, a): a["name"] for a in AGENTS}
            agent_reports = {}
            for fut in as_completed(futures):
                name = futures[fut]
                agent_reports[name] = fut.result()
                print(f"  ✓ {name} finished")

        # --- Synthesis: head coach on Opus with adaptive thinking ---
        report_text = "\n\n".join(
            f"=== REPORT FROM {a['name'].upper()} ===\n{agent_reports[a['name']]}"
            for a in AGENTS
        )

        synthesis_blocks = [{
            "type": "text",
            "text": (
                "You are the HEAD GOLF COACH. Three specialist coaches each reviewed the same golf swing "
                "independently — one focused on the body, one on the club and hands, one on tempo and rhythm. "
                "Their full reports are below. The swing frames are also attached so you can verify their claims.\n\n"
                + shared_intro + "\n\n" + report_text
            ),
        }]
        synthesis_blocks.extend(image_blocks)
        synthesis_blocks.append({
            "type": "text",
            "text": (
                "Write the final coaching report in plain, everyday language anyone can understand — even "
                "someone who has never played golf. Explain any golf term in brackets immediately. "
                "Use exactly this structure:\n\n"
                "## Quick Summary\n"
                "2-3 sentences on the overall picture, honest and encouraging.\n\n"
                "## Where the Coaches Agree\n"
                "List the findings that two or more specialists independently spotted. These are the most reliable "
                "observations. For each, note which coaches saw it.\n\n"
                "## Where the Coaches See It Differently\n"
                "Point out anything one specialist flagged that the others didn't mention or saw differently, and give "
                "your judgment as head coach on who is right and why. If they fully agree on everything, say so.\n\n"
                "## The Complete Issue List\n"
                "Combine ALL issues from all three coaches into one master list — every single one, not just the top few. "
                "Group them by severity:\n"
                "### Major Issues (fix these first)\n"
                "### Moderate Issues\n"
                "### Minor Issues (polish for later)\n"
                "For each issue: a short plain title, one sentence on what's happening, which coach(es) spotted it, "
                "and one simple drill or tip to fix it.\n\n"
                "## What's Working Well\n"
                "Everything the coaches praised, combined. Be specific about why each thing matters.\n\n"
                "## Suggested Practice Order\n"
                "A short numbered list: which issue to work on first, second, third, and so on — and why that order. "
                "Fixing one thing often fixes others downstream; use that logic."
            ),
        })

        print(f"Running head-coach synthesis [{SYNTHESIS_MODEL}]...")
        synthesis_response = ai_client.messages.create(
            model=SYNTHESIS_MODEL,
            max_tokens=6000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": synthesis_blocks}],
        )
        analysis_text = "".join(b.text for b in synthesis_response.content if b.type == "text")

    except anthropic.APIStatusError as exc:
        traceback.print_exc()
        if using_free:
            refund_free_analysis(ip)
        status = exc.status_code
        if status in (502, 503, 504):
            msg = f"Anthropic's servers returned a temporary {status} error. Please wait a moment and try again."
        elif status == 529:
            msg = "Anthropic's API is overloaded right now. Please wait a minute and try again."
        elif status == 401:
            msg = "Invalid API key — please check it and try again."
        elif status == 429:
            msg = "Rate limit hit. Please wait a moment and try again."
        else:
            import re
            raw = str(exc.message or exc.body or "")
            msg = re.sub(r"<[^>]+>", "", raw).strip() or f"API error {status}"
        return jsonify({"error": msg}), 502
    except anthropic.APITimeoutError:
        traceback.print_exc()
        if using_free:
            refund_free_analysis(ip)
        return jsonify({"error": "Request timed out. Try again — the server may be busy."}), 504
    except anthropic.APIConnectionError as exc:
        traceback.print_exc()
        if using_free:
            refund_free_analysis(ip)
        return jsonify({"error": f"Connection error: {str(exc)[:200]}"}), 502

    if not analysis_text.strip():
        if using_free:
            refund_free_analysis(ip)
        return jsonify({"error": "Claude returned an empty response. Please try again."}), 502

    # Return annotated frames so the browser can display them
    frames_payload = []
    for b64 in front_frames:
        frames_payload.append({"view": "Front", "data": b64})
    for b64 in back_frames:
        frames_payload.append({"view": "Back", "data": b64})

    return jsonify({
        "analysis": analysis_text,
        "frames": frames_payload,
        "agent_reports": [
            {"name": a["name"], "report": agent_reports[a["name"]]} for a in AGENTS
        ],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
