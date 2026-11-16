import math
import random
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

# ---------------- settings ----------------
CAMERA_INDEX = 0            # try 1 if you get a black screen
CW, CH = 1280, 720          # game resolution (16:9). Fullscreen scales it to your screen.
CAM_W, CAM_H = 640, 480     # what the webcam captures. LOW = fast. (try 1280, 720 for sharper video)
TW, TH = 384, 216           # bumped from 256x144 for much more reliable fast-swipe detection
START_LIVES = 5
SLICE_SPEED = 550           # min fingertip speed (pixels/second at 540p) to count as a slice
TARGET_FPS = 60             # game frame cap
SHOW_STATS = True           # shows CAM / TRACK speed so you can see what is slow
UI = CH / 540.0             # UI scale factor

# BGR colours. skin = outside, flesh = inside, rind = thin layer under the skin
FRUITS = {
    "apple":      dict(skin=(35, 35, 220),   rim=(15, 15, 130),  flesh=(200, 240, 250), rind=(35, 35, 220),   scale=1.0,  ay=1.0,  gloss=0.75),
    "orange":     dict(skin=(0, 140, 255),   rim=(0, 80, 180),   flesh=(30, 165, 255),  rind=(210, 240, 255), scale=1.0,  ay=1.0,  gloss=0.35),
    "watermelon": dict(skin=(65, 155, 35),   rim=(25, 85, 15),   flesh=(75, 65, 240),   rind=(210, 240, 210), scale=1.3,  ay=1.0,  gloss=0.55),
    "lemon":      dict(skin=(0, 225, 250),   rim=(0, 155, 185),  flesh=(155, 245, 250), rind=(205, 245, 250), scale=0.95, ay=0.78, gloss=0.45),
    "kiwi":       dict(skin=(45, 105, 140),  rim=(20, 55, 85),   flesh=(80, 210, 130),  rind=(60, 130, 160),  scale=0.85, ay=1.0,  gloss=0.20),
}

FRAME_TIME = 1.0 / TARGET_FPS

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


# ---------------- One-Euro Filter (Jitter-free + Zero-lag tracking) ----------------
class LowPassFilter:
    def __init__(self, alpha, initval=0.0):
        self.y = self.s = initval
        self.initialized = False

    def filter(self, value, alpha):
        if not self.initialized:
            self.s = value
            self.initialized = True
        else:
            self.s = alpha * value + (1.0 - alpha) * self.s
        self.y = value
        return self.s


class OneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=0.04, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x = LowPassFilter(self._alpha(min_cutoff, 1 / 60))
        self.dx = LowPassFilter(self._alpha(d_cutoff, 1 / 60))
        self.last_time = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-5))

    def reset(self):
        self.x.initialized = False
        self.dx.initialized = False
        self.last_time = None

    def __call__(self, x, t):
        if self.last_time is None:
            dt = 1.0 / 60.0
        else:
            dt = max(t - self.last_time, 1e-5)
        self.last_time = t

        dx = 0.0 if not self.x.initialized else (x - self.x.s) / dt
        edx = self.dx.filter(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x.filter(x, self._alpha(cutoff, dt))


# ---------------- camera thread ----------------
class Camera:
    def __init__(self, index):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame, self.fid, self.running = None, 0, True
        self.on_small = None
        self.fps, self._t = 0.0, time.perf_counter()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, raw = self.cap.read()
            if ok:
                t = time.perf_counter()
                self.fps = self.fps * 0.9 + (1.0 / max(t - self._t, 1e-4)) * 0.1
                self._t = t
                bg, small = prepare(raw)
                with self.lock:
                    self.frame, self.fid = bg, self.fid + 1
                cb = self.on_small
                if cb:
                    cb(small)
            else:
                time.sleep(0.01)

    def latest(self):
        with self.lock:
            return self.frame, self.fid

    def wait_first(self, timeout=5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest()[0] is not None:
                return True
            time.sleep(0.05)
        return False

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ---------------- hand tracking thread ----------------
class HandTracker:
    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")
    MODEL_FILE = "hand_landmarker.task"

    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions")
        self.last_ts = 0
        if self.legacy:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.55, min_tracking_confidence=0.45)
        else:
            import os
            import urllib.request
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.MODEL_FILE)
            if not os.path.exists(path):
                print("Downloading hand tracking model (one time, ~8 MB)...")
                urllib.request.urlretrieve(self.MODEL_URL, path)
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.55,
                min_tracking_confidence=0.45,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

        self.filter_x = OneEuroFilter(min_cutoff=1.5, beta=0.05)
        self.filter_y = OneEuroFilter(min_cutoff=1.5, beta=0.05)
        self.missed_frames = 0

        self.lock = threading.Lock()
        self.event = threading.Event()
        self.inbox = None
        self.tip, self.seq, self.t = None, 0, 0.0
        self.fps, self.ms, self._t = 0.0, 0.0, time.perf_counter()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, small_bgr):
        with self.lock:
            self.inbox = small_bgr
        self.event.set()

    def get(self):
        with self.lock:
            return self.tip, self.seq, self.t

    def _extract_tip(self, landmarks):
        p8 = landmarks[8]
        p7 = landmarks[7]
        x = p8.x * 0.88 + p7.x * 0.12
        y = p8.y * 0.88 + p7.y * 0.12
        return x, y

    def _detect(self, rgb):
        if self.legacy:
            rgb.flags.writeable = False
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                return self._extract_tip(res.multi_hand_landmarks[0].landmark)
            return None
        ts = max(self.last_ts + 1, int(time.perf_counter() * 1000))
        self.last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect_for_video(img, ts)
        if res.hand_landmarks:
            return self._extract_tip(res.hand_landmarks[0])
        return None

    def _loop(self):
        while self.running:
            self.event.wait()
            self.event.clear()
            with self.lock:
                small, self.inbox = self.inbox, None
            if small is None:
                continue
            t0 = time.perf_counter()
            raw_tip = self._detect(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            t1 = time.perf_counter()
            self.ms = self.ms * 0.8 + (t1 - t0) * 1000 * 0.2
            self.fps = self.fps * 0.9 + (1.0 / max(t1 - self._t, 1e-4)) * 0.1
            self._t = t1

            if raw_tip is not None:
                self.missed_frames = 0
                fx = self.filter_x(raw_tip[0], t1)
                fy = self.filter_y(raw_tip[1], t1)
                tip = (max(0.0, min(1.0, fx)), max(0.0, min(1.0, fy)))
            else:
                self.missed_frames += 1
                if self.missed_frames > 3:
                    self.filter_x.reset()
                    self.filter_y.reset()
                    tip = None
                else:
                    with self.lock:
                        tip = self.tip

            with self.lock:
                self.tip, self.t, self.seq = tip, t1, self.seq + 1


# ---------------- 3D sprite shading & drawing ----------------
def _ell(img, mask, c, axes, col, m=True, ang=0):
    axes = (max(1, int(axes[0])), max(1, int(axes[1])))
    c = (int(c[0]), int(c[1]))
    cv2.ellipse(img, c, axes, ang, 0, 360, col, -1, cv2.LINE_AA)
    if m:
        cv2.ellipse(mask, c, axes, ang, 0, 360, 255, -1, cv2.LINE_AA)


def _line(img, mask, p1, p2, col, t, m=True):
    p1 = (int(p1[0]), int(p1[1])); p2 = (int(p2[0]), int(p2[1]))
    cv2.line(img, p1, p2, col, t, cv2.LINE_AA)
    if m:
        cv2.line(mask, p1, p2, 255, t, cv2.LINE_AA)


def _canvas(R):
    N = int(R * 3.2)
    return np.zeros((N, N, 3), np.uint8), np.zeros((N, N), np.uint8), N, N // 2


def _apply_3d_shading(img, mask, cx, cy, rx, ry, gloss=0.5, is_flat_cut=False):
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    nx = (x - cx) / max(rx, 1.0)
    ny = (y - cy) / max(ry, 1.0)
    r2 = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

    lx, ly, lz = -0.45, -0.55, 0.70
    ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

    if is_flat_cut:
        shade = 0.78 + 0.28 * ndotl
        spec = (ndotl ** 14) * 45 * gloss
    else:
        ambient = 0.38
        diffuse = 0.72 * ndotl
        rim = np.clip(1.0 - nz, 0.0, 1.0) ** 2.5 * np.clip(-ndotl, 0.0, 1.0) * 0.25
        shade = ambient + diffuse + rim

        hx, hy, hz = lx, ly, lz + 1.0
        hlen = math.sqrt(hx * hx + hy * hy + hz * hz)
        hx, hy, hz = hx / hlen, hy / hlen, hz / hlen
        ndoth = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)
        spec = (ndoth ** 24) * 140 * gloss

    shaded = img.astype(np.float32) * shade[..., None] + spec[..., None]
    shaded = np.clip(shaded, 0, 255).astype(np.uint8)

    body = (r2 <= 1.02) & (mask > 0)
    img[body] = shaded[body]


def make_whole(kind, base_R):
    sp = FRUITS[kind]
    R = int(base_R * sp["scale"]); ay = sp["ay"]
    img, mask, N, c = _canvas(R)
    C = (c, c)
    rnd = random.Random(kind)

    _ell(img, mask, C, (R, R * ay), sp["rim"])
    _ell(img, mask, C, (R - 2, R * ay - 2), sp["skin"], m=False)

    if kind == "apple":
        for dx in (-0.45, -0.2, 0.15, 0.45):
            cv2.ellipse(img, (int(c + dx * R * 0.3), c), (int(R * 0.65), int(R * 0.96)),
                        0, 0, 360, (25, 25, 190), 2, cv2.LINE_AA)
        _apply_3d_shading(img, mask, c, c, R, R * ay, gloss=sp["gloss"])
        _ell(img, mask, (c, c - R * .78), (R * .24, R * .09), (20, 20, 110), m=False)
        _line(img, mask, (c, c - R * .80), (c + R * .12, c - R * 1.28), (30, 60, 100), 5)
        _ell(img, mask, (c + R * .38, c - R * 1.12), (R * .32, R * .14), (45, 175, 45), ang=-25)
        _ell(img, mask, (c + R * .35, c - R * 1.15), (R * .24, R * .06), (90, 225, 90), m=False, ang=-25)

    elif kind == "orange":
        for _ in range(75):
            a, d = rnd.uniform(0, 6.28), rnd.uniform(0, R * .92)
            cv2.circle(img, (int(c + d * math.cos(a)), int(c + d * math.sin(a))), 1, (0, 95, 210), -1)
        _apply_3d_shading(img, mask, c, c, R, R * ay, gloss=sp["gloss"])
        _ell(img, mask, (c, c - R * .88), (R * .18, R * .09), (35, 145, 35))

    elif kind == "watermelon":
        for f in (0.28, 0.55, 0.8, 0.95):
            cv2.ellipse(img, C, (max(1, int(R * f)), R - 2), 0, 0, 360, (25, 90, 15), 5, cv2.LINE_AA)
        _apply_3d_shading(img, mask, c, c, R, R * ay, gloss=sp["gloss"])
        _line(img, mask, (c, c - R + 2), (c + 3, c - R * 1.22), (30, 80, 20), 6)

    elif kind == "lemon":
        _ell(img, mask, (c - R * .98, c), (R * .14, R * .11), sp["skin"])
        _ell(img, mask, (c + R * .98, c), (R * .14, R * .11), sp["skin"])
        _apply_3d_shading(img, mask, c, c, R * 1.05, R * ay, gloss=sp["gloss"])

    elif kind == "kiwi":
        for _ in range(110):
            a, d = rnd.uniform(0, 6.28), rnd.uniform(0, R * .92)
            cv2.circle(img, (int(c + d * math.cos(a)), int(c + d * math.sin(a))), 1, (75, 145, 185), -1)
        _apply_3d_shading(img, mask, c, c, R, R * ay, gloss=sp["gloss"])

    return img, mask, N, R


def make_sliced(kind, base_R):
    sp = FRUITS[kind]
    R = int(base_R * sp["scale"]); ay = sp["ay"]
    img, mask, N, c = _canvas(R)
    C = (c, c)
    rnd = random.Random(kind + "cut")

    _ell(img, mask, (c + 3, c + 3), (R, R * ay), sp["rim"])
    _ell(img, mask, C, (R, R * ay), sp["skin"])
    _ell(img, mask, C, (R * .91, R * ay * .91), sp["rind"], m=False)
    _ell(img, mask, C, (R * .83, R * ay * .83), sp["flesh"], m=False)

    if kind in ("orange", "lemon"):
        for k in range(8):
            a = k * math.pi / 4
            _line(img, mask, C, (c + R * .8 * math.cos(a), c + R * .8 * ay * math.sin(a)), sp["rind"], 2, m=False)
        _ell(img, mask, C, (R * .09, R * .09), sp["rind"], m=False)
    elif kind == "apple":
        _ell(img, mask, C, (R * .17, R * .3), (150, 205, 225), m=False)
        for dx in (-1, 1):
            _ell(img, mask, (c + dx * R * .06, c), (3, 6), (20, 40, 80), m=False)
    elif kind == "watermelon":
        for _ in range(10):
            a, d = rnd.uniform(0, 6.28), rnd.uniform(R * .15, R * .62)
            _ell(img, mask, (c + d * math.cos(a), c + d * math.sin(a)), (3, 7), (25, 25, 25), m=False, ang=rnd.randint(0, 180))
    elif kind == "kiwi":
        for k in range(14):
            a = k * 2 * math.pi / 14
            cv2.circle(img, (int(c + R * .42 * math.cos(a)), int(c + R * .42 * math.sin(a))), 2, (20, 20, 20), -1)
        _ell(img, mask, C, (R * .16, R * .16), (225, 245, 235), m=False)

    _apply_3d_shading(img, mask, c, c, R, R * ay, gloss=0.65, is_flat_cut=True)

    li, lm = img.copy(), mask.copy()
    ri, rm = img.copy(), mask.copy()
    lm[:, c:] = 0; li[:, c:] = 0
    rm[:, :c] = 0; ri[:, :c] = 0
    return (li, lm, N, R), (ri, rm, N, R)


def make_bomb(base_R):
    R = int(base_R * 0.95)
    img, mask, N, c = _canvas(R)
    _ell(img, mask, (c, c), (R, R), (45, 45, 50))
    cv2.line(img, (c - int(R * 0.85), c + R // 6), (c + int(R * 0.85), c + R // 6), (0, 0, 210), 4, cv2.LINE_AA)
    _apply_3d_shading(img, mask, c, c, R, R, gloss=0.85)

    _line(img, mask, (c + R * .55, c - R * .75), (c + R * .95, c - R * 1.2), (40, 90, 140), 5)
    _ell(img, mask, (c + R * .97, c - R * 1.25), (10, 10), (0, 120, 255))
    _ell(img, mask, (c + R * .97, c - R * 1.25), (5, 5), (180, 255, 255), m=False)
    return img, mask, N, R


_ANG_STEPS = 72
_rot_cache = {}


def _rotated(spr, ang):
    idx = int(round(ang * _ANG_STEPS / (2 * math.pi))) % _ANG_STEPS
    key = (id(spr[0]), idx)
    e = _rot_cache.get(key, 0)
    if e != 0:
        return e
    img, mask, N, _ = spr
    c = N / 2
    a = idx * 2 * math.pi / _ANG_STEPS
    cs, sn = math.cos(a), math.sin(a)
    M = np.float32([[cs, -sn, c - cs * c + sn * c],
                    [sn,  cs, c - sn * c - cs * c]])
    ri = cv2.warpAffine(img, M, (N, N))
    rm = cv2.warpAffine(mask, M, (N, N))
    bx, by, bw, bh = cv2.boundingRect(rm)
    if bw == 0 or bh == 0:
        e = None
    else:
        rm3 = cv2.merge([rm, rm, rm])
        pre = cv2.multiply(ri, rm3, scale=1 / 255)
        inv = cv2.bitwise_not(rm3)
        sh_inv = 255 - (rm3 // 3)
        e = (np.ascontiguousarray(pre[by:by + bh, bx:bx + bw]),
             np.ascontiguousarray(inv[by:by + bh, bx:bx + bw]),
             np.ascontiguousarray(sh_inv[by:by + bh, bx:bx + bw]), bx, by)
    _rot_cache[key] = e
    return e


def blit(frame, spr, cx, cy, ang, shadow_offset=10):
    e = _rotated(spr, ang)
    if e is None:
        return
    pre, inv, sh_inv, bx, by = e
    ph, pw = pre.shape[:2]

    if shadow_offset > 0:
        sx0 = int(cx - spr[2] / 2) + bx + shadow_offset
        sy0 = int(cy - spr[2] / 2) + by + shadow_offset
        sfx1, sfy1 = max(sx0, 0), max(sy0, 0)
        sfx2, sfy2 = min(sx0 + pw, frame.shape[1]), min(sy0 + ph, frame.shape[0])
        if sfx2 > sfx1 and sfy2 > sfy1:
            ssx1, ssy1 = sfx1 - sx0, sfy1 - sy0
            ssx2, ssy2 = ssx1 + (sfx2 - sfx1), ssy1 + (sfy2 - sfy1)
            sroi = frame[sfy1:sfy2, sfx1:sfx2]
            cv2.multiply(sroi, sh_inv[ssy1:ssy2, ssx1:ssx2], dst=sroi, scale=1 / 255)

    x0, y0 = int(cx - spr[2] / 2) + bx, int(cy - spr[2] / 2) + by
    fx1, fy1 = max(x0, 0), max(y0, 0)
    fx2, fy2 = min(x0 + pw, frame.shape[1]), min(y0 + ph, frame.shape[0])
    if fx2 <= fx1 or fy2 <= fy1:
        return
    sx1, sy1 = fx1 - x0, fy1 - y0
    sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)
    roi = frame[fy1:fy2, fx1:fx2]
    cv2.multiply(roi, inv[sy1:sy2, sx1:sx2], dst=roi, scale=1 / 255)
    cv2.add(roi, pre[sy1:sy2, sx1:sx2], dst=roi)


# ---------------- helpers ----------------
def drop(lst, obj):
    for i, o in enumerate(lst):
        if o is obj:
            del lst[i]
            return


def seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def text(frame, msg, pos, scale=1.0, color=(255, 255, 255), th=2):
    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, msg, pos, f, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(frame, msg, pos, f, scale, color, th, cv2.LINE_AA)


def centered(frame, msg, y, scale=1.0, color=(255, 255, 255), th=2):
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    text(frame, msg, ((frame.shape[1] - tw) // 2, y), scale, color, th)


# ---------------- game ----------------
def prepare(raw):
    f = raw
    h, w = f.shape[:2]
    if w * 9 > h * 16:
        nw = h * 16 // 9
        x0 = (w - nw) // 2
        f = f[:, x0:x0 + nw]
    elif w * 9 < h * 16:
        nh = w * 9 // 16
        y0 = (h - nh) // 2
        f = f[y0:y0 + nh]
    f = cv2.flip(f, 1)
    small = cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA)
    if f.shape[1] != CW or f.shape[0] != CH:
        f = cv2.resize(f, (CW, CH))
    return f, small


def main():
    cam = Camera(CAMERA_INDEX)
    if not cam.wait_first():
        print("Could not open the camera. Try changing CAMERA_INDEX at the top of the file.")
        return
    tracker = HandTracker()
    cam.on_small = tracker.submit

    win = "ExoSlash"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fullscreen = True
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    w, h = CW, CH
    s = UI
    G = 0.1125 * s
    base_R = int(h * 0.075)
    whole = {k: make_whole(k, base_R) for k in FRUITS}
    cut = {k: make_sliced(k, base_R) for k in FRUITS}
    bomb_spr = make_bomb(base_R)

    fruits, halves, juice = [], [], []
    trail = deque()
    prev, prev_t = None, 0.0
    score, best, lives = 0, 0, START_LIVES
    state = "HOME"          # "HOME", "PLAY", or "OVER"
    pending_action = None   # ("HOME" or "QUIT", trigger_time) for slice feedback delay
    timer, flash = 30.0, 0.0
    bg, last_seq = np.zeros((h, w, 3), np.uint8), 0
    red = np.full((h, w, 3), (0, 0, 255), np.uint8)
    tint = np.full((h, w, 3), (22, 16, 12), np.uint8)
    fps, t_last, tip = 0.0, time.perf_counter(), None
    vel = (0.0, 0.0)

    # Interactive sliceable menu targets
    start_fruit_pos = (w // 2, int(h * 0.56))
    start_fruit_r = int(base_R * 1.25)

    over_y = int(h * 0.64)
    over_r = int(base_R * 1.05)
    over_buttons = [
        dict(id="RESTART", label="RESTART", kind="orange", pos=(int(w * 0.27), over_y), ring=(0, 180, 255)),
        dict(id="HOME",    label="HOME",    kind="apple",  pos=(int(w * 0.50), over_y), ring=(255, 210, 60)),
        dict(id="QUIT",    label="QUIT",    kind="bomb",   pos=(int(w * 0.73), over_y), ring=(60, 60, 255)),
    ]

    def spawn_slice_fx(kind, pos, ux, uy):
        if kind == "bomb":
            for _ in range(30):
                a, v = random.uniform(0, 6.28), random.uniform(1.5, 6.5) * s
                juice.append(dict(x=pos[0], y=pos[1], vx=v * math.cos(a), vy=v * math.sin(a),
                                  life=40, color=(0, random.randint(100, 220), 255),
                                  size=random.randint(3, 8)))
        else:
            ang = math.atan2(-ux, uy)
            left, right = cut[kind]
            for spr, side in ((left, 1), (right, -1)):
                halves.append(dict(spr=spr, x=pos[0], y=pos[1],
                                   vx=side * -uy * 2.2 * s,
                                   vy=-2.5 * s + side * ux * 1.8 * s,
                                   ang=ang, spin=side * 0.03, r=base_R))
            fl = FRUITS[kind]["flesh"]
            for _ in range(22):
                a, v = random.uniform(0, 6.28), random.uniform(1.5, 5.5) * s
                juice.append(dict(x=pos[0], y=pos[1], vx=v * math.cos(a), vy=v * math.sin(a),
                                  life=44, color=fl, size=random.randint(3, 7)))

    def start_game():
        nonlocal state, score, lives, timer, flash, pending_action
        fruits.clear()
        halves.clear()
        juice.clear()
        score = 0
        lives = START_LIVES
        timer = 25.0
        flash = 0.0
        pending_action = None
        state = "PLAY"

    def spawn():
        is_bomb = random.random() < min(0.10 + score * 0.002, 0.25)
        kind = "bomb" if is_bomb else random.choice(list(FRUITS))
        spr = bomb_spr if is_bomb else whole[kind]
        x = random.uniform(0.15 * w, 0.85 * w)
        vy = -math.sqrt(2 * G * random.uniform(0.55, 0.85) * h)
        air_time = 2 * -vy / G
        vx = (random.uniform(0.2, 0.8) * w - x) / air_time
        return dict(kind=kind, spr=spr, r=spr[3], x=x, y=h + spr[3], vx=vx, vy=vy,
                    ang=random.uniform(0, 6.28), spin=random.uniform(-0.04, 0.04))

    def check_slice(p1, p2, speed):
        nonlocal score, lives, flash, pending_action
        if speed <= SLICE_SPEED * s:
            return
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return
        ux, uy = dx / dist, dy / dist

        # HOME screen: slice center apple to START
        if state == "HOME":
            if seg_dist(start_fruit_pos, p1, p2) < start_fruit_r * 1.15:
                start_game()
                spawn_slice_fx("apple", start_fruit_pos, ux, uy)
            return

        # GAME OVER screen: slice RESTART / HOME / QUIT targets
        if state == "OVER" and pending_action is None:
            for btn in over_buttons:
                if seg_dist(btn["pos"], p1, p2) < over_r * 1.15:
                    if btn["id"] == "RESTART":
                        start_game()
                        spawn_slice_fx(btn["kind"], btn["pos"], ux, uy)
                    elif btn["id"] == "HOME":
                        spawn_slice_fx(btn["kind"], btn["pos"], ux, uy)
                        pending_action = ("HOME", time.perf_counter() + 0.38)
                    elif btn["id"] == "QUIT":
                        flash = 25
                        spawn_slice_fx(btn["kind"], btn["pos"], ux, uy)
                        pending_action = ("QUIT", time.perf_counter() + 0.35)
                    return
            return

        if state != "PLAY":
            return

        for f in fruits[:]:
            if seg_dist((f["x"], f["y"]), p1, p2) < f["r"] * 1.02:
                drop(fruits, f)
                if f["kind"] == "bomb":
                    lives -= 1
                    flash = 20
                    spawn_slice_fx("bomb", (f["x"], f["y"]), ux, uy)
                else:
                    score += 1
                    ang = math.atan2(-ux, uy)
                    left, right = cut[f["kind"]]
                    for spr, side in ((left, 1), (right, -1)):
                        halves.append(dict(spr=spr, x=f["x"], y=f["y"],
                                           vx=f["vx"] + side * -uy * 1.6 * s,
                                           vy=f["vy"] * 0.5 + side * ux * 1.6 * s,
                                           ang=ang, spin=side * random.uniform(0.015, 0.04), r=f["r"]))
                    fl = FRUITS[f["kind"]]["flesh"]
                    for _ in range(18):
                        a, v = random.uniform(0, 6.28), random.uniform(1.2, 5.0) * s
                        juice.append(dict(x=f["x"], y=f["y"], vx=v * math.cos(a), vy=v * math.sin(a),
                                          life=44, color=fl, size=random.randint(2, 6)))

    def draw_hud_brackets(frame, col=(255, 210, 60)):
        m_pad, b_len = int(28 * s), int(35 * s)
        for cx_b, cy_b, dx_b, dy_b in (
            (m_pad, m_pad, 1, 1),
            (w - m_pad, m_pad, -1, 1),
            (m_pad, h - m_pad, 1, -1),
            (w - m_pad, h - m_pad, -1, -1),
        ):
            cv2.line(frame, (cx_b, cy_b), (cx_b + dx_b * b_len, cy_b), col, 2, cv2.LINE_AA)
            cv2.line(frame, (cx_b, cy_b), (cx_b, cy_b + dy_b * b_len), col, 2, cv2.LINE_AA)

    while True:
        now = time.perf_counter()
        dt = now - t_last
        t_last = now
        k = min(dt * 60.0, 3.0)
        fps = fps * 0.9 + (1.0 / max(dt, 1e-4)) * 0.1

        # Handle delayed HOME / QUIT transition after slice animation plays
        if pending_action is not None and now >= pending_action[1]:
            act = pending_action[0]
            pending_action = None
            if act == "HOME":
                fruits.clear(); halves.clear(); juice.clear()
                state = "HOME"
            elif act == "QUIT":
                break

        cam_frame, _ = cam.latest()
        if cam_frame is not None:
            bg = cam_frame
        frame = bg.copy()

        # Dim background slightly on menu screens so UI pops cleanly
        if state in ("HOME", "OVER"):
            cv2.addWeighted(frame, 0.55, tint, 0.45, 0, dst=frame)

        # newest hand result
        tip_n, seq, t_res = tracker.get()
        tip = (int(tip_n[0] * w), int(tip_n[1] * h)) if tip_n else None
        if seq != last_seq:
            last_seq = seq
            if tip:
                trail.append((tip[0], tip[1], now))
                if prev and t_res > prev_t:
                    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
                    dist = math.hypot(dx, dy)
                    dt_track = max(t_res - prev_t, 1e-3)
                    speed = dist / dt_track
                    vel = (dx / dt_track, dy / dt_track)
                    check_slice(prev, tip, speed)
                prev, prev_t = tip, t_res
            else:
                vel = (0.0, 0.0)
                prev = None
                trail.clear()

        # ---------------- HOME SCREEN ----------------
        if state == "HOME":
            draw_hud_brackets(frame, (255, 210, 60))

            centered(frame, "EXOSLASH", int(h * 0.24), 2.3 * s, (255, 255, 255), 5)
            centered(frame, "OPTICAL KINETIC BLADE SYSTEM", int(h * 0.31), 0.62 * s, (255, 210, 60), 2)

            pulse = 1.0 + 0.06 * math.sin(now * 4.5)
            ring_r = int(start_fruit_r * 1.35 * pulse)
            cv2.circle(frame, start_fruit_pos, ring_r, (255, 210, 60), 2, cv2.LINE_AA)
            cv2.circle(frame, start_fruit_pos, int(ring_r * 1.12), (180, 140, 40), 1, cv2.LINE_AA)
            bob_y = start_fruit_pos[1] + math.sin(now * 3.0) * 6 * s
            blit(frame, whole["apple"], start_fruit_pos[0], bob_y, now * 0.7, shadow_offset=int(14 * s))

            centered(frame, "SLICE THE APPLE TO START", int(h * 0.77), 0.9 * s, (255, 255, 255), 2)
            if tip is not None:
                centered(frame, "[ HAND SENSOR: LOCKED ]", int(h * 0.84), 0.62 * s, (80, 255, 120), 2)
            else:
                centered(frame, "[ SHOW INDEX FINGER TO CAMERA ]", int(h * 0.84), 0.62 * s, (0, 220, 255), 2)
            centered(frame, "F = Fullscreen   |   Q = Quit", int(h * 0.91), 0.52 * s, (190, 190, 190), 1)

        # ---------------- GAME OVER SCREEN ----------------
        if state == "OVER":
            draw_hud_brackets(frame, (60, 60, 230))
            centered(frame, "SYSTEM OVERLOAD // GAME OVER", int(h * 0.23), 1.45 * s, (60, 60, 255), 4)
            centered(frame, f"FINAL SCORE: {score}     |     BEST: {best}", int(h * 0.33), 0.95 * s, (255, 255, 255), 2)
            centered(frame, "SLICE A TARGET TO SELECT", int(h * 0.42), 0.65 * s, (255, 210, 60), 2)

            if pending_action is None:
                pulse = 1.0 + 0.05 * math.sin(now * 5.0)
                for idx, btn in enumerate(over_buttons):
                    bx, by = btn["pos"]
                    ring_r = int(over_r * 1.32 * pulse)
                    cv2.circle(frame, (bx, by), ring_r, btn["ring"], 2, cv2.LINE_AA)
                    cv2.circle(frame, (bx, by), int(ring_r * 1.12), btn["ring"], 1, cv2.LINE_AA)

                    bob_y = by + math.sin(now * 3.2 + idx * 1.4) * 5 * s
                    spr = bomb_spr if btn["kind"] == "bomb" else whole[btn["kind"]]
                    blit(frame, spr, bx, bob_y, now * (0.7 if idx % 2 == 0 else -0.7), shadow_offset=int(12 * s))

                    (tw, _), _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, 2)
                    text(frame, btn["label"], (bx - tw // 2, by + int(over_r * 1.85)), 0.8 * s, (255, 255, 255), 2)

        # ---------------- GAMEPLAY & PHYSICS ----------------
        if state == "PLAY":
            timer -= k
            if timer <= 0:
                for _ in range(random.choice([1, 1, 2, 3])):
                    fruits.append(spawn())
                timer = max(36.0, 100.0 - score * 0.7)

        for f in fruits[:]:
            f["vy"] += G * k
            f["x"] += f["vx"] * k
            f["y"] += f["vy"] * k
            f["ang"] += f["spin"] * k
            if f["y"] - f["r"] > h and f["vy"] > 0:
                drop(fruits, f)
                if f["kind"] != "bomb" and state == "PLAY":
                    lives -= 1
        for p in halves[:]:
            p["vy"] += G * k
            p["x"] += p["vx"] * k
            p["y"] += p["vy"] * k
            p["ang"] += p["spin"] * k
            if p["y"] - p["r"] * 2 > h:
                drop(halves, p)
        for j in juice[:]:
            j["vy"] += G * 0.6 * k
            j["x"] += j["vx"] * k
            j["y"] += j["vy"] * k
            j["life"] -= k
            if j["life"] <= 0:
                drop(juice, j)

        if lives <= 0 and state == "PLAY":
            state = "OVER"
            best = max(best, score)

        # draw active 3D sprites & juice
        for f in fruits:
            blit(frame, f["spr"], f["x"], f["y"], f["ang"], shadow_offset=int(12 * s))
        for p in halves:
            blit(frame, p["spr"], p["x"], p["y"], p["ang"], shadow_offset=int(8 * s))
        for j in juice:
            jx, jy = int(j["x"]), int(j["y"])
            cv2.circle(frame, (jx, jy), j["size"], j["color"], -1, cv2.LINE_AA)
            if j["size"] >= 4:
                cv2.circle(frame, (jx - 1, jy - 1), 1, (255, 255, 255), -1)

        # draw blade trail & predictive tip (active in all states!)
        while trail and now - trail[0][2] > 0.22:
            trail.popleft()
        pts = list(trail)
        for i in range(1, len(pts)):
            life = 1 - (now - pts[i][2]) / 0.22
            t = max(2, int(11 * s * life))
            a, b = (pts[i - 1][0], pts[i - 1][1]), (pts[i][0], pts[i][1])
            cv2.line(frame, a, b, (255, 210, 60), t + 6, cv2.LINE_AA)
            cv2.line(frame, a, b, (255, 255, 255), t, cv2.LINE_AA)
        if tip:
            lead = min(max(now - t_res, 0.0) + 0.025, 0.065)
            head = (int(tip[0] + vel[0] * lead), int(tip[1] + vel[1] * lead))
            if trail:
                last_pt = (trail[-1][0], trail[-1][1])
                cv2.line(frame, last_pt, head, (255, 255, 255), max(2, int(5 * s)), cv2.LINE_AA)
                check_slice(last_pt, head, math.hypot(vel[0], vel[1]))
            cv2.circle(frame, head, int(7 * s), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, head, int(10 * s), (255, 210, 60), 2, cv2.LINE_AA)

        if flash > 0:
            cv2.addWeighted(red, 0.35, frame, 0.65, 0, dst=frame)
            flash -= k

        # HUD during PLAY
        if state == "PLAY":
            text(frame, f"Score: {score}", (int(20 * s), int(40 * s)), 1.0 * s)
            text(frame, f"Best: {max(best, score)}", (int(20 * s), int(75 * s)), 0.7 * s, (0, 255, 255))
            for i in range(START_LIVES):
                col = (0, 0, 255) if i < lives else (90, 90, 90)
                c = (w - int(40 * s) - i * int(42 * s), int(38 * s))
                cv2.circle(frame, c, int(15 * s), col, -1, cv2.LINE_AA)
                cv2.circle(frame, c, int(15 * s), (255, 255, 255), 2, cv2.LINE_AA)

        if SHOW_STATS:
            text(frame, f"CAM {cam.fps:.0f} fps | TRACK {tracker.fps:.0f} fps ({tracker.ms:.0f} ms) | GAME {fps:.0f} fps",
                 (int(20 * s), h - int(15 * s)), 0.5 * s, (200, 200, 200), 1)
        if tip is None and state == "PLAY":
            centered(frame, "Show your hand and swipe your index finger!", h - int(25 * s), 0.8 * s, (0, 255, 255))

        cv2.imshow(win, frame)
        wait = max(1, int((FRAME_TIME - (time.perf_counter() - now)) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if state == "HOME" and key in (ord(" "), 13):
            start_game()
        if key == ord("r") and state in ("PLAY", "OVER"):
            start_game()
        if key == ord("h") and state == "OVER":
            fruits.clear(); halves.clear(); juice.clear()
            state = "HOME"

    tracker.running = False
    tracker.event.set()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()