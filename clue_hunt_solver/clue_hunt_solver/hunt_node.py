#!/usr/bin/env python3

import hashlib
import math
import re
import traceback

import cv2
import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)

import tf2_ros
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:  # pragma: no cover
    zbar_decode = None

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_rot(x, y, z, w):
    xx, yy, zz = x * x, y * y, z * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)]])


def unit(v, fallback=None):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return fallback
    return np.asarray(v, dtype=float) / n


# ----------------------------------------------------------------------------------------------
class HuntNode(Node):

    def __init__(self):
        super().__init__('hunt_node')
        self._declare_params()
        P = self.p

        # ---------------- ROS interfaces ----------------
        self.bridge = CvBridge()
        self.create_subscription(Image, P('image_topic'), self.image_cb, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, P('camera_info_topic'), self.caminfo_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, P('scan_topic'), self.scan_cb, qos_profile_sensor_data)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, P('map_topic'), self.map_cb, map_qos)

        self.clue_pub = self.create_publisher(String, '/hunt/clues', 10)
        self.board_pub = self.create_publisher(String, '/hunt/boards', 10)
        self.treasure_pub = self.create_publisher(PoseStamped, '/hunt/treasure', 10)
        self.status_pub = self.create_publisher(String, '/leader/status', 10)
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, '/spin')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- perception setup ----------------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        try:
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        except Exception:
            pass
        self.aruco_detector = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ---------------- sensor cache ----------------
        self.last_img = None
        self.frame_seq = 0
        self._cached_seq = -1
        self._cached = None
        self.img_shape = None
        self.K = None
        self.D = np.zeros(5)
        self.scan = None
        self.map_msg = None
        self.map_arr = None
        self.warned = set()
        self.last_pose = None

        # ---------------- hunt state ----------------
        self.expected_id = int(P('first_board_id'))
        self.expected_tokens = {str(P('first_token')).upper()}
        self.rejected = []            # map xy of decoys / look-alikes
        self.last_board = None
        self.pillars = {}             # COLOR -> np.array([x, y])
        self.plan = None
        self.search_center = None
        self.ring_radius = float(P('ring_radius'))
        self.ring_done = False
        self.vantages = []
        self.visited = []

        # scan state
        self.scan_mode = 'board'
        self.scan_color = None
        self.scan_steps = 0
        self.pillar_samples = []
        self.dwell_t0 = 0.0
        self.dwell_seq0 = 0
        self.dwell_evals = 0
        self.dwell_seen = False
        self.last_eval_seq = -1
        self.max_scan_steps = max(1, int(round(360.0 / float(P('search_step_deg')))))

        # approach / read state
        self.board_est = None
        self.approach_goals = []
        self.approach_i = 0
        self.approach_cycles = 0
        self.read_poses = []
        self.read_t0 = 0.0
        self.read_seq0 = 0

        # target nav
        self.target_goals = []
        self.target_i = 0
        self.treasure = None
        self.treasure_tries = 0

        # action status
        self.nav_status = 'idle'      # idle | pending | active | ok | fail
        self.nav_token = 0
        self.nav_handle = None
        self.nav_t0 = 0.0
        self.spin_status = 'idle'
        self.spin_token = 0
        self.spin_handle = None
        self.spin_t0 = 0.0

        self.state = 'INIT'
        self.state_t0 = 0.0
        self.init_t0 = None
        self.init_pose_sent = False
        self.resume = ('board', None)

        self.create_timer(0.1, self.tick)
        self.get_logger().info(
            f'Hunt node up. Waiting for board {self.expected_id}, token {sorted(self.expected_tokens)}')

    # ==========================================================================================
    # parameters / utilities
    # ==========================================================================================
    def _declare_params(self):
        d = self.declare_parameter
        d('image_topic', '/camera/image_raw')
        d('camera_info_topic', '/camera/camera_info')
        d('scan_topic', '/scan')
        d('map_topic', '/map')
        d('map_frame', 'map')
        d('base_frame', 'base_link')
        d('marker_size', 0.24)
        d('first_board_id', 1)
        d('first_token', '7196')
        d('camera_optical', 'auto')         # auto | true | false
        d('camera_hfov', 1.047)             # used only if CameraInfo never arrives
        d('scan_yaw_offset', 0.0)
        d('search_step_deg', 45.0)
        d('dwell_sec', 0.8)
        d('standoffs', [1.2, 1.6, 0.9])
        d('target_standoffs', [1.5, 2.2, 1.0])
        d('ring_radius', 1.8)
        d('ring_points', 8)
        d('hop_distance', 2.5)
        d('pillar_radius', 0.15)
        d('pillar_distance', 1.8)
        d('rel_y_sign', 1.0)                # flip to -1.0 if REL's y axis turns out mirrored
        d('nav_timeout', 90.0)
        d('read_timeout', 6.0)
        d('max_marker_range', 7.0)
        d('publish_initial_pose', True)
        d('init_pose_wait', 10.0)
        self.p = lambda name: self.get_parameter(name).value

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            self.get_logger().warning(msg)

    def set_state(self, name):
        if name != self.state:
            self.get_logger().info(f'[state] {self.state} -> {name}')
        self.state = name
        self.state_t0 = self.now()
        m = String()
        m.data = name
        self.status_pub.publish(m)

    # ==========================================================================================
    # callbacks
    # ==========================================================================================
    def image_cb(self, msg):
        self.last_img = msg
        self.frame_seq += 1

    def caminfo_cb(self, msg):
        if msg.k[0] > 1.0:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            d = np.array(msg.d, dtype=np.float64)
            self.D = d if d.size else np.zeros(5)

    def scan_cb(self, msg):
        self.scan = msg

    def map_cb(self, msg):
        self.map_msg = msg
        self.map_arr = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

    def get_frame(self):
        if self.last_img is None:
            return None
        if self._cached_seq != self.frame_seq:
            try:
                bgr = self.bridge.imgmsg_to_cv2(self.last_img, desired_encoding='bgr8')
            except Exception as e:
                self.warn_once('cvb', f'cv_bridge failed: {e}')
                return None
            self._cached = (bgr, self.last_img.header.frame_id)
            self._cached_seq = self.frame_seq
            self.img_shape = bgr.shape
        return self._cached

    def intrinsics(self):
        if self.K is not None:
            return self.K, self.D
        if self.img_shape is None:
            return None, None
        h, w = self.img_shape[:2]
        f = (w / 2.0) / math.tan(float(self.p('camera_hfov')) / 2.0)
        self.warn_once('k', 'No CameraInfo received - using camera_hfov fallback intrinsics')
        return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]]), np.zeros(5)

    # ==========================================================================================
    # TF
    # ==========================================================================================
    def tf_matrix(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except Exception:
            return None
        q = t.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
        T[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        return T

    def robot_pose(self):
        T = self.tf_matrix(self.p('map_frame'), self.p('base_frame'))
        if T is None:
            return self.last_pose
        self.last_pose = (T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0]))
        return self.last_pose

    def cam_to_map(self, frame_id):
        T = self.tf_matrix(self.p('map_frame'), frame_id)
        if T is None:
            self.warn_once('camtf', f'TF map->{frame_id} missing; assuming camera sits at '
                                    f'{self.p("base_frame")} looking forward')
            T = self.tf_matrix(self.p('map_frame'), self.p('base_frame'))
        return T

    def camera_is_optical(self, frame_id):
        mode = str(self.p('camera_optical')).lower()
        if mode == 'true':
            return True
        if mode == 'false':
            return False
        return 'optical' in frame_id.lower()

    # ==========================================================================================
    # map helpers
    # ==========================================================================================
    def is_free(self, x, y, margin=0.3):
        m = self.map_msg
        if m is None or self.map_arr is None:
            return True
        res = m.info.resolution
        i = int((x - m.info.origin.position.x) / res)
        j = int((y - m.info.origin.position.y) / res)
        k = max(1, int(margin / res))
        h, w = self.map_arr.shape
        if i - k < 0 or j - k < 0 or i + k >= w or j + k >= h:
            return False
        win = self.map_arr[j - k:j + k + 1, i - k:i + k + 1]
        return bool(np.all((win >= 0) & (win < 50)))

    # ==========================================================================================
    # LiDAR helpers
    # ==========================================================================================
    def lidar_range(self, rel_angle, window):
        s = self.scan
        if s is None or s.angle_increment == 0.0:
            return None
        n = len(s.ranges)
        full = n * abs(s.angle_increment) >= 2 * math.pi - 0.2
        if full:
            rel_angle = s.angle_min + ((rel_angle - s.angle_min) % (2 * math.pi))
        idx = int(round((rel_angle - s.angle_min) / s.angle_increment))
        w = max(1, int(window / abs(s.angle_increment)))
        vals = []
        for k in range(-w, w + 1):
            j = (idx + k) % n if full else idx + k
            if 0 <= j < n:
                r = s.ranges[j]
                if math.isfinite(r) and s.range_min <= r <= s.range_max:
                    vals.append(r)
        return min(vals) if vals else None

    def best_open_direction(self, pose):
        """Pick a hop goal toward free space that we have not visited yet."""
        s = self.scan
        if s is None:
            return None
        rx, ry, ryaw = pose
        r = np.array(s.ranges, dtype=float)
        r = np.where(np.isfinite(r), r, s.range_max)
        r = np.clip(r, 0.0, 6.0)
        n = len(r)
        inc = s.angle_increment
        k = max(1, int(math.radians(20) / abs(inc)))
        padded = np.r_[r[-k:], r, r[:k]]
        smooth = np.convolve(padded, np.ones(2 * k + 1) / (2 * k + 1), mode='valid')
        hop = float(self.p('hop_distance'))
        best, best_score = None, -1e9
        for idx in range(0, n, max(1, int(math.radians(10) / abs(inc)))):
            rng = float(smooth[idx])
            ang = wrap(s.angle_min + idx * inc + float(self.p('scan_yaw_offset')) + ryaw)
            dist = min(hop, 0.6 * float(r[idx]), 0.8 * rng)
            if dist < 0.6:
                continue
            gx, gy = rx + dist * math.cos(ang), ry + dist * math.sin(ang)
            score = rng
            if any(math.hypot(gx - vx, gy - vy) < 1.5 for vx, vy in self.visited):
                score -= 3.0
            if not self.is_free(gx, gy, 0.25):
                score -= 5.0
            if score > best_score:
                best_score, best = score, (gx, gy, ang)
        return best

    # ==========================================================================================
    # action clients
    # ==========================================================================================
    def send_nav(self, x, y, yaw):
        if not self.nav_client.server_is_ready():
            self.get_logger().error('NavigateToPose server not ready')
            self.nav_status = 'fail'
            return
        self.nav_token += 1
        tk = self.nav_token
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self.p('map_frame')
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        goal.pose = ps
        self.nav_status = 'pending'
        self.nav_t0 = self.now()
        self.nav_handle = None
        self.get_logger().info(f'Nav goal -> x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f}deg')
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(lambda f, tk=tk: self._nav_accepted(f, tk))

    def _nav_accepted(self, fut, tk):
        try:
            gh = fut.result()
        except Exception as e:
            self.get_logger().error(f'nav goal error: {e}')
            if tk == self.nav_token:
                self.nav_status = 'fail'
            return
        if tk != self.nav_token:
            if gh is not None and gh.accepted:
                gh.cancel_goal_async()
            return
        if not gh.accepted:
            self.nav_status = 'fail'
            return
        self.nav_handle = gh
        self.nav_status = 'active'
        gh.get_result_async().add_done_callback(lambda f, tk=tk: self._nav_result(f, tk))

    def _nav_result(self, fut, tk):
        if tk != self.nav_token:
            return
        try:
            st = fut.result().status
        except Exception:
            st = GoalStatus.STATUS_ABORTED
        self.nav_status = 'ok' if st == GoalStatus.STATUS_SUCCEEDED else 'fail'
        self.nav_handle = None

    def cancel_nav(self):
        self.nav_token += 1
        if self.nav_handle is not None:
            try:
                self.nav_handle.cancel_goal_async()
            except Exception:
                pass
        self.nav_handle = None
        self.nav_status = 'idle'

    def nav_poll(self):
        if self.nav_status in ('pending', 'active') and \
                self.now() - self.nav_t0 > float(self.p('nav_timeout')):
            self.get_logger().warning('Nav2 goal timed out - cancelling')
            self.cancel_nav()
            self.nav_status = 'fail'
        return self.nav_status

    def send_spin(self, angle):
        if not self.spin_client.server_is_ready():
            self.warn_once('spin', 'Spin server not ready (is behavior_server running?)')
            self.spin_status = 'fail'
            return
        self.spin_token += 1
        tk = self.spin_token
        goal = Spin.Goal()
        goal.target_yaw = float(angle)
        goal.time_allowance.sec = 12
        self.spin_status = 'pending'
        self.spin_t0 = self.now()
        self.spin_handle = None
        fut = self.spin_client.send_goal_async(goal)
        fut.add_done_callback(lambda f, tk=tk: self._spin_accepted(f, tk))

    def _spin_accepted(self, fut, tk):
        try:
            gh = fut.result()
        except Exception:
            if tk == self.spin_token:
                self.spin_status = 'fail'
            return
        if tk != self.spin_token:
            return
        if not gh.accepted:
            self.spin_status = 'fail'
            return
        self.spin_handle = gh
        self.spin_status = 'active'
        gh.get_result_async().add_done_callback(lambda f, tk=tk: self._spin_result(f, tk))

    def _spin_result(self, fut, tk):
        if tk != self.spin_token:
            return
        try:
            st = fut.result().status
        except Exception:
            st = GoalStatus.STATUS_ABORTED
        self.spin_status = 'ok' if st == GoalStatus.STATUS_SUCCEEDED else 'fail'
        self.spin_handle = None

    def cancel_spin(self):
        self.spin_token += 1
        if self.spin_handle is not None:
            try:
                self.spin_handle.cancel_goal_async()
            except Exception:
                pass
        self.spin_handle = None
        self.spin_status = 'idle'

    # ==========================================================================================
    # vision: ArUco, QR, solvePnP
    # ==========================================================================================
    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict,
                                                      parameters=self.aruco_params)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2).astype(np.float64)) for i, c in zip(ids.flatten(), corners)]

    def marker_pose_map(self, corners, frame_id):
        """solvePnP -> camera frame -> map frame.  Returns pos, normal (toward the camera), dist."""
        K, D = self.intrinsics()
        if K is None:
            return None
        s = float(self.p('marker_size')) / 2.0
        obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, D,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        n = R[:, 2].copy()
        if t[2] <= 0.1 or np.linalg.norm(t) > float(self.p('max_marker_range')):
            return None
        if float(np.dot(n, t)) > 0.0:       # make the normal point toward the camera
            n = -n
        if not self.camera_is_optical(frame_id):
            t = R_OPT_TO_LINK @ t
            n = R_OPT_TO_LINK @ n
        T = self.cam_to_map(frame_id)
        if T is None:
            return None
        pos = T[:3, :3] @ t + T[:3, 3]
        nm = T[:3, :3] @ n
        return {'pos': pos, 'normal': nm, 'dist': float(np.linalg.norm(t))}

    # ---- QR ---------------------------------------------------------------------------------
    def _decode_gray(self, gray):
        out = []
        if zbar_decode is not None:
            try:
                for o in zbar_decode(gray):
                    text = o.data.decode('utf-8', errors='ignore').strip()
                    if text:
                        x, y, w, h = o.rect
                        out.append((text, np.array([x + w / 2.0, y + h / 2.0])))
            except Exception as e:
                self.warn_once('zbar', f'pyzbar failed: {e}')
        if not out:
            try:
                det = cv2.QRCodeDetector()
                ok, infos, pts, _ = det.detectAndDecodeMulti(gray)
                if ok:
                    for text, p in zip(infos, pts):
                        if text:
                            out.append((text.strip(), p.mean(axis=0)))
            except Exception:
                pass
        return out

    def decode_qrs(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = self._decode_gray(gray)
        if not res:
            up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            res = [(t, c / 2.0) for t, c in self._decode_gray(up)]
        return res

    def decode_beside_marker(self, frame, corners):
        """Fallback: crop the region to the right of the marker, upscale, decode."""
        c = corners
        u = c[1] - c[0]
        v = c[3] - c[0]
        m = c.mean(axis=0)
        pts = [m + u * a + v * b for a in (0.5, 5.0) for b in (-2.0, 2.5)]
        pts = np.array(pts)
        h, w = frame.shape[:2]
        x0, y0 = np.maximum(pts.min(axis=0).astype(int), 0)
        x1, y1 = np.minimum(pts.max(axis=0).astype(int), [w - 1, h - 1])
        if x1 - x0 < 20 or y1 - y0 < 20:
            return []
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        return [(t, c0 / 3.0 + np.array([x0, y0])) for t, c0 in self._decode_gray(crop)]

    def match_qr(self, corners, qrs, frame):
        centre = corners.mean(axis=0)
        side = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
        cands = [(float(np.linalg.norm(c - centre)), t) for t, c in qrs]
        cands = [x for x in cands if x[0] < 8.0 * side]
        if not cands:
            cands = [(float(np.linalg.norm(c - centre)), t)
                     for t, c in self.decode_beside_marker(frame, corners)]
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        return cands[0][1]

    # ---- one-stop board observation -----------------------------------------------------------
    def is_rejected(self, pos):
        return any(math.hypot(pos[0] - r[0], pos[1] - r[1]) < 0.6 for r in self.rejected)

    def observe_board(self, frame, frame_id, decode_qr):
        mine = [c for (i, c) in self.detect_markers(frame) if i == self.expected_id]
        if not mine:
            return []
        qrs = self.decode_qrs(frame) if decode_qr else []
        obs = []
        for c in mine:
            pose = self.marker_pose_map(c, frame_id)
            if pose is None or self.is_rejected(pose['pos']):
                continue
            text = self.match_qr(c, qrs, frame) if decode_qr else None
            obs.append({'pose': pose, 'text': text})
        return obs

    @staticmethod
    def pose_median(poses):
        pos = np.median(np.array([p['pos'] for p in poses]), axis=0)
        n = np.mean(np.array([unit(p['normal'][:2], np.zeros(2)) for p in poses]), axis=0)
        return {'pos': pos, 'normal': np.array([n[0], n[1], 0.0]),
                'dist': float(np.median([p['dist'] for p in poses]))}

    # ---- clue validation ----------------------------------------------------------------------
    @staticmethod
    def token_candidates(text):
        """Plausible readings of 'SHA-1 of the previous clue'. The matching one gets logged."""
        m = CLUE_RE.match(text.strip())
        variants = {'full': text.strip()}
        if m:
            variants['command_only'] = m.group(3).strip()
            variants['without_HUNT_prefix'] = text.strip()[len('HUNT:'):]
            variants['id_token_command_no_space'] = text.strip().replace(' ', '')
        out = {}
        for name, s in variants.items():
            out[hashlib.sha1(s.encode('utf-8')).hexdigest().upper()[:4]] = name
        return out

    def validate_clue(self, text):
        m = CLUE_RE.match(text.strip())
        if m is None:
            return 'ignore', None
        board_id, token, cmd = int(m.group(1)), m.group(2).upper(), m.group(3).strip()
        if board_id != self.expected_id:
            return 'decoy', None
        if token not in self.expected_tokens:
            return 'lookalike', None
        return 'ok', (board_id, token, cmd)

    # ---- pillars ------------------------------------------------------------------------------
    def detect_pillar(self, frame, color):
        rngs = HSV_RANGES.get(color.upper())
        if rngs is None:
            self.warn_once('col' + color, f'Unknown pillar colour {color}')
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in rngs:
            mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < 400:
            return None
        x, y, w, h = cv2.boundingRect(best)
        if h < 25:
            return None
        return {'cx': x + w / 2.0, 'cy': y + h / 2.0, 'w': w, 'h': h}

    def estimate_pillar_pos(self, det, frame_id):
        K, _ = self.intrinsics()
        T = self.cam_to_map(frame_id)
        rp = self.robot_pose()
        if K is None or T is None or rp is None:
            return None
        ray = np.array([(det['cx'] - K[0, 2]) / K[0, 0], (det['cy'] - K[1, 2]) / K[1, 1], 1.0])
        if not self.camera_is_optical(frame_id):
            ray = R_OPT_TO_LINK @ ray
        d = T[:3, :3] @ ray
        bearing = math.atan2(d[1], d[0])
        rx, ry, ryaw = rp
        rel = wrap(bearing - ryaw - float(self.p('scan_yaw_offset')))
        r = self.lidar_range(rel, math.radians(3.0))
        if r is None:
            return None
        dist = r + float(self.p('pillar_radius'))
        return np.array([rx + dist * math.cos(bearing), ry + dist * math.sin(bearing)])

    # ==========================================================================================
    # main loop
    # ==========================================================================================
    def tick(self):
        try:
            getattr(self, 'st_' + self.state.lower())()
        except Exception:
            self.get_logger().error('tick failed:\n' + traceback.format_exc())

    # ---------------- INIT -------------------------------------------------------------------
    def st_init(self):
        t = self.now()
        if self.init_t0 is None:
            self.init_t0 = t
        pose = self.robot_pose()
        have = {
            'camera frames': self.frame_seq > 0,
            'NavigateToPose server': self.nav_client.server_is_ready(),
            'Spin server': self.spin_client.server_is_ready(),
            'TF map->base': pose is not None,
        }
        if pose is None and self.p('publish_initial_pose') and not self.init_pose_sent \
                and t - self.init_t0 > float(self.p('init_pose_wait')):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = self.p('map_frame')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose.orientation.w = 1.0
            msg.pose.covariance[0] = msg.pose.covariance[7] = 0.05
            msg.pose.covariance[35] = 0.05
            self.init_pub.publish(msg)
            self.init_pose_sent = True
            self.get_logger().info('Published /initialpose at the origin (robot spawns there)')
        if all(have.values()):
            self.get_logger().info('All systems ready - starting hunt')
            self.begin_scan('board')
            return
        if t - getattr(self, '_last_init_log', -99.0) > 5.0:
            self._last_init_log = t
            missing = [k for k, v in have.items() if not v]
            self.get_logger().info(f'Waiting for: {missing}')

    # ---------------- SCAN -------------------------------------------------------------------
    def begin_scan(self, mode, color=None):
        self.cancel_spin()
        self.scan_mode, self.scan_color = mode, color
        self.resume = (mode, color)
        self.scan_steps = 0
        if mode == 'pillar':
            self.pillar_samples = []
            self.get_logger().info(f'Scanning for {color} pillar')
        else:
            self.get_logger().info(f'Scanning for board {self.expected_id}')
        self.enter_dwell()

    def enter_dwell(self):
        self.dwell_t0 = self.now()
        self.dwell_seq0 = self.frame_seq
        self.dwell_evals = 0
        self.dwell_seen = False
        self.set_state('SCAN_DWELL')

    def st_scan_dwell(self):
        fr = self.get_frame()
        elapsed = self.now() - self.dwell_t0
        fresh = self.frame_seq >= self.dwell_seq0 + 3
        if fr is not None and fresh and self.frame_seq != self.last_eval_seq:
            self.last_eval_seq = self.frame_seq
            self.dwell_evals += 1
            frame, fid = fr

            obs = self.observe_board(frame, fid, decode_qr=False)
            if obs:
                best = min(obs, key=lambda o: o['pose']['dist'])
                p = best['pose']['pos']
                self.get_logger().info(
                    f'Board {self.expected_id} marker seen at ({p[0]:.2f},{p[1]:.2f}), '
                    f'{best["pose"]["dist"]:.2f} m away')
                self.start_approach(best['pose'])
                return

            if self.scan_mode == 'pillar':
                det = self.detect_pillar(frame, self.scan_color)
                if det is not None:
                    self.dwell_seen = True
                    pos = self.estimate_pillar_pos(det, fid)
                    if pos is not None:
                        self.pillar_samples.append(pos)

        limit = float(self.p('dwell_sec'))
        done = (elapsed >= limit and self.dwell_evals >= 2) or elapsed >= limit + 3.0
        if not done:
            return
        if self.scan_mode == 'pillar':
            if len(self.pillar_samples) >= 3 and self.finish_pillar():
                return
            if self.dwell_seen and elapsed < 3.0:
                return                      # keep collecting samples
        self.scan_steps += 1
        if self.scan_steps >= self.max_scan_steps:
            self.on_scan_exhausted()
            return
        self.send_spin(math.radians(float(self.p('search_step_deg'))))
        self.set_state('SCAN_SPIN')

    def st_scan_spin(self):
        if self.spin_status in ('ok', 'fail') or self.now() - self.spin_t0 > 15.0:
            if self.spin_status in ('pending', 'active'):
                self.cancel_spin()
            self.enter_dwell()

    def finish_pillar(self):
        arr = np.array(self.pillar_samples)
        med = np.median(arr, axis=0)
        close = arr[np.linalg.norm(arr - med, axis=1) < 0.5]
        if len(close) < 3:
            self.pillar_samples = []
            return False
        pos = np.median(close, axis=0)
        self.pillars[self.scan_color] = pos
        self.get_logger().info(f'{self.scan_color} pillar located at ({pos[0]:.2f}, {pos[1]:.2f})')
        self.advance_plan()
        return True

    def on_scan_exhausted(self):
        self.get_logger().warning('Full 360 scan found nothing')
        if self.scan_mode == 'board':
            if self.vantages:
                self.visit_next_vantage()
                return
            if self.search_center is not None and not self.ring_done:
                self.build_ring()
                self.visit_next_vantage()
                return
        self.start_hop()

    # ---------------- vantage ring / hop ------------------------------------------------------
    def build_ring(self):
        self.ring_done = True
        cx, cy = self.search_center
        rp = self.robot_pose() or (0.0, 0.0, 0.0)
        n = int(self.p('ring_points'))
        pts = []
        for k in range(n):
            a = 2 * math.pi * k / n
            x, y = cx + self.ring_radius * math.cos(a), cy + self.ring_radius * math.sin(a)
            if self.is_free(x, y):
                pts.append((x, y, wrap(a + math.pi)))
        pts.sort(key=lambda p: math.hypot(p[0] - rp[0], p[1] - rp[1]))
        self.vantages = pts
        self.get_logger().info(f'Built {len(pts)} vantage points around '
                               f'({cx:.2f},{cy:.2f}) r={self.ring_radius:.2f}')

    def visit_next_vantage(self):
        if not self.vantages:
            self.start_hop()
            return
        x, y, yaw = self.vantages.pop(0)
        self.visited.append((x, y))
        self.send_nav(x, y, yaw)
        self.set_state('VANTAGE_NAV')

    def st_vantage_nav(self):
        if self.nav_poll() in ('ok', 'fail'):
            self.begin_scan('board')

    def start_hop(self):
        rp = self.robot_pose()
        hop = self.best_open_direction(rp) if rp is not None else None
        if hop is None:
            self.get_logger().warning('No hop direction available - rescanning in place')
            self.begin_scan(*self.resume)
            return
        gx, gy, yaw = hop
        self.visited.append((gx, gy))
        self.get_logger().info('Hopping toward open space to look from a new spot')
        self.send_nav(gx, gy, yaw)
        self.set_state('HOP_NAV')

    def st_hop_nav(self):
        if self.nav_poll() in ('ok', 'fail'):
            self.begin_scan(*self.resume)

    # ---------------- APPROACH ----------------------------------------------------------------
    def standoff_goals(self, pose):
        pos = pose['pos'][:2]
        rp = self.robot_pose() or (0.0, 0.0, 0.0)
        n = unit(pose['normal'][:2])
        if n is None or np.linalg.norm(pose['normal'][:2]) < 0.25:
            n = unit(np.array(rp[:2]) - pos, np.array([1.0, 0.0]))
        goals = []
        for d in self.p('standoffs'):
            g = pos + n * float(d)
            goals.append((float(g[0]), float(g[1]), math.atan2(-n[1], -n[0])))
        free = [g for g in goals if self.is_free(g[0], g[1])]
        return free + [g for g in goals if g not in free]

    def start_approach(self, pose):
        self.board_est = pose
        self.approach_goals = self.standoff_goals(pose)
        self.approach_i = 0
        self.go_approach_goal()

    def go_approach_goal(self):
        if self.approach_i >= len(self.approach_goals):
            self.approach_cycles += 1
            self.get_logger().warning('All stand-off goals failed for this board')
            if self.approach_cycles >= 3 and self.board_est is not None:
                self.rejected.append(self.board_est['pos'][:2].copy())
                self.get_logger().warning('Treating this marker as unreachable - ignoring it')
                self.approach_cycles = 0
            self.begin_scan('board')
            return
        x, y, yaw = self.approach_goals[self.approach_i]
        self.send_nav(x, y, yaw)
        self.set_state('APPROACH')

    def st_approach(self):
        s = self.nav_poll()
        if s == 'ok':
            self.enter_read()
        elif s == 'fail':
            self.approach_i += 1
            self.go_approach_goal()

    # ---------------- READ --------------------------------------------------------------------
    def enter_read(self):
        self.read_t0 = self.now()
        self.read_seq0 = self.frame_seq
        self.read_poses = []
        self.last_eval_seq = -1
        self.set_state('READ')

    def st_read(self):
        fr = self.get_frame()
        if fr is not None and self.frame_seq >= self.read_seq0 + 3 \
                and self.frame_seq != self.last_eval_seq:
            self.last_eval_seq = self.frame_seq
            frame, fid = fr
            for o in self.observe_board(frame, fid, decode_qr=True):
                self.read_poses.append(o['pose'])
                self.read_poses = self.read_poses[-8:]
                if not o['text']:
                    continue
                verdict, parsed = self.validate_clue(o['text'])
                if verdict == 'ok':
                    self.accept_clue(o['text'], parsed)
                    return
                if verdict in ('decoy', 'lookalike'):
                    self.get_logger().warning(
                        f'Rejected {verdict}: "{o["text"]}" (expected id {self.expected_id}, '
                        f'tokens {sorted(self.expected_tokens)})')
                    self.rejected.append(o['pose']['pos'][:2].copy())
                    self.begin_scan('board')
                    return
        if self.now() - self.read_t0 > float(self.p('read_timeout')):
            if not self.read_poses:
                self.get_logger().warning('Marker lost during read - rescanning')
                self.begin_scan('board')
                return
            self.get_logger().warning('QR not readable from this stand-off - trying another')
            self.board_est = self.pose_median(self.read_poses)
            self.approach_i += 1
            self.go_approach_goal()

    def accept_clue(self, text, parsed):
        board_id, token, cmd = parsed
        text = text.strip()
        bp = self.pose_median(self.read_poses) if self.read_poses else self.board_est
        self.approach_cycles = 0

        self.get_logger().info('=' * 56)
        self.get_logger().info(f'VALID CLUE: {text}')
        self.get_logger().info(f'board={board_id} token={token} command="{cmd}"')
        self.get_logger().info(f'board pose in map: ({bp["pos"][0]:.2f}, {bp["pos"][1]:.2f})')
        self.get_logger().info('=' * 56)

        m = String()
        m.data = text
        self.clue_pub.publish(m)
        b = String()
        b.data = f'{board_id} {bp["pos"][0]:.3f} {bp["pos"][1]:.3f}'
        self.board_pub.publish(b)

        cands = self.token_candidates(text)
        self.expected_tokens = set(cands.keys())
        self.get_logger().info(f'Token variants for next board: {cands}')
        self.expected_id = board_id + 1
        self.last_board = bp
        self.search_center = None
        self.ring_done = False
        self.vantages = []
        self.dispatch(cmd, bp)

    # ---------------- command dispatch --------------------------------------------------------
    @staticmethod
    def floats(s):
        return [float(x) for x in NUM_RE.findall(s)]

    def rel_to_map(self, bp, dx, dy):
        n = unit(bp['normal'][:2], np.array([1.0, 0.0]))
        right = np.array([-n[1], n[0]]) * float(self.p('rel_y_sign'))
        return bp['pos'][:2] + dx * n + dy * right

    def dispatch(self, cmd, bp):
        up = cmd.upper().strip()
        nums = self.floats(cmd)
        words = [w for w in re.findall(r'[A-Za-z]+', up) if w not in ('AND', 'TO')]
        if up.startswith('TREASURE'):
            if len(nums) >= 2:
                self.go_treasure(self.rel_to_map(bp, nums[0], nums[1]))
                return
        elif up.startswith('GOTO'):
            if len(nums) >= 2:
                self.get_logger().info(f'GOTO ({nums[0]:.2f}, {nums[1]:.2f})')
                self.go_to_target(np.array([nums[0], nums[1]]))
                return
        elif up.startswith('REL'):
            if len(nums) >= 2:
                tgt = self.rel_to_map(bp, nums[0], nums[1])
                self.get_logger().info(f'REL -> map ({tgt[0]:.2f}, {tgt[1]:.2f})')
                self.go_to_target(tgt)
                return
        elif up.startswith('PILLAR'):
            if len(words) >= 2:
                dist = nums[0] if nums else float(self.p('pillar_distance'))
                self.plan = {'type': 'PILLAR', 'colors': [words[1]], 'dist': dist}
                self.advance_plan()
                return
        elif up.startswith('BETWEEN'):
            if len(words) >= 3:
                t = nums[0] if nums else 0.5
                self.plan = {'type': 'BETWEEN', 'colors': [words[1], words[2]], 't': t}
                self.advance_plan()
                return
        self.get_logger().error(f'Could not interpret command "{cmd}" - scanning around instead')
        self.begin_scan('board')

    def advance_plan(self):
        p = self.plan
        for c in p['colors']:
            if c not in self.pillars:
                self.begin_scan('pillar', c)
                return
        if p['type'] == 'PILLAR':
            center = self.pillars[p['colors'][0]]
            self.search_center = center.copy()
            self.ring_radius = float(p['dist'])
            self.ring_done = False
            self.build_ring()
            self.visit_next_vantage()
        else:
            a, b = self.pillars[p['colors'][0]], self.pillars[p['colors'][1]]
            tgt = a + p['t'] * (b - a)
            self.get_logger().info(f'BETWEEN target ({tgt[0]:.2f}, {tgt[1]:.2f})')
            self.go_to_target(tgt)

    # ---------------- go to an approximate board location -------------------------------------
    def go_to_target(self, target):
        self.search_center = np.array(target, dtype=float)
        self.ring_radius = float(self.p('ring_radius'))
        self.ring_done = False
        self.vantages = []
        rp = self.robot_pose()
        if rp is None:
            self.begin_scan('board')
            return
        v = self.search_center - np.array(rp[:2])
        d = float(np.linalg.norm(v))
        u = unit(v, np.array([math.cos(rp[2]), math.sin(rp[2])]))
        goals = []
        for s in self.p('target_standoffs'):
            if d <= float(s) + 0.3:
                continue
            g = self.search_center - u * float(s)
            goals.append((float(g[0]), float(g[1]), math.atan2(u[1], u[0])))
        free = [g for g in goals if self.is_free(g[0], g[1])]
        self.target_goals = free + [g for g in goals if g not in free]
        self.target_i = 0
        if not self.target_goals:
            self.get_logger().info('Already near the target - scanning')
            self.begin_scan('board')
            return
        self.go_target_goal()

    def go_target_goal(self):
        if self.target_i >= len(self.target_goals):
            self.begin_scan('board')
            return
        self.send_nav(*self.target_goals[self.target_i])
        self.set_state('TARGET_NAV')

    def st_target_nav(self):
        s = self.nav_poll()
        if s == 'ok':
            self.begin_scan('board')
        elif s == 'fail':
            self.target_i += 1
            self.go_target_goal()

    # ---------------- treasure ----------------------------------------------------------------
    def go_treasure(self, xy):
        self.treasure = np.array(xy, dtype=float)
        self.treasure_tries = 0
        ps = PoseStamped()
        ps.header.frame_id = self.p('map_frame')
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(xy[0]), float(xy[1])
        ps.pose.orientation.w = 1.0
        self.treasure_pub.publish(ps)
        self.get_logger().info(f'TREASURE at map ({xy[0]:.2f}, {xy[1]:.2f}) - driving onto it')
        self.drive_treasure()

    def drive_treasure(self):
        rp = self.robot_pose() or (0.0, 0.0, 0.0)
        v = self.treasure - np.array(rp[:2])
        yaw = math.atan2(v[1], v[0])
        # nudge the goal a little toward the robot on retries (in case the point is inside an inflated zone)
        back = 0.12 * self.treasure_tries
        g = self.treasure - unit(v, np.array([1.0, 0.0])) * back
        self.send_nav(float(g[0]), float(g[1]), yaw)
        self.set_state('TREASURE_NAV')

    def st_treasure_nav(self):
        s = self.nav_poll()
        if s == 'ok':
            self.get_logger().info('*' * 56)
            self.get_logger().info('TREASURE REACHED')
            self.get_logger().info('*' * 56)
            self.set_state('DONE')
        elif s == 'fail':
            self.treasure_tries += 1
            if self.treasure_tries > 4:
                self.get_logger().error('Could not reach the treasure point')
                self.set_state('DONE')
            else:
                self.drive_treasure()

    def st_done(self):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = HuntNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
