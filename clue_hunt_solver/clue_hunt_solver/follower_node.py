#!/usr/bin/env python3
"""
follower_node.py -- camera-only follower for the Clue Chain Hunt (Inter IIT Bootcamp, Phase 2)

Uses ONLY the follower's own camera (ArUco id 49 on the leader's back, DICT_4X4_50, 0.12 m) and its
own wheel odometry.  No LiDAR, no map, no leader data.

Why this version exists
-----------------------
The previous follower stopped printing / following after the leader started scanning: the leader
spins in place, the tag on its back turns away, the camera loses it and nothing brought it back.
This node has an explicit state machine:

    TRACK   tag visible.  Range/bearing -> (v, w).  Forward speed is scaled by cos(bearing)^2 and
            is zero when the heading error is large, so it never drives forward at full speed while
            turning at full speed (the old failure).  A feed-forward on the filtered range rate
            keeps up with a moving leader.  Backs up slowly if the leader comes closer than d_min.
    HOLD    tag lost.  Stop (never keep a stale command), keep the camera pointed at the leader's
            last known position (dead-reckoned in odom) and wait: a spinning leader shows its tag
            again within one sweep.
    SEARCH  still nothing after hold_sec.  Rotate in place through 360 deg (never drive blind),
            then go back to HOLD facing the last known position, alternate the turn direction, repeat
            forever.

Topic names are auto-discovered (anything with 'follower' in the name) unless you pass them as
parameters.  Check with:   ros2 topic list | grep follower
Output: /follower/cmd_vel (geometry_msgs/Twist)
"""

import math
import traceback

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class FollowerNode(Node):

    def __init__(self):
        super().__init__('follower_node')
        d = self.declare_parameter
        d('image_topic', '')               # '' = auto-discover
        d('camera_info_topic', '')
        d('odom_topic', '')
        d('cmd_topic', '/follower/cmd_vel')
        d('tag_id', 49)
        d('tag_size', 0.12)
        d('cam_forward_offset', 0.0)       # camera ahead of the robot centre (m)
        d('camera_hfov', 1.047)            # fallback if CameraInfo never arrives
        d('d_des', 1.2)                    # target following distance (band is 0.6 - 2.0 m)
        d('d_deadband', 0.15)
        d('d_min', 0.65)                   # closer than this -> back up slowly
        d('allow_reverse', True)
        d('v_max', 0.6)
        d('v_rev', 0.12)
        d('w_max', 1.0)
        d('kv', 0.8)
        d('kw', 1.6)
        d('k_ff', 0.7)                     # range-rate feed-forward
        d('acc_lim', 0.6)                  # m/s^2 speed-up
        d('dec_lim', 1.5)                  # m/s^2 slow-down
        d('turn_in_place', 0.7)            # rad: above this heading error, rotate only
        d('lost_after', 0.5)               # s without a detection -> lost
        d('hold_sec', 8.0)
        d('search_w', 0.5)
        d('min_perimeter_rate', 0.02)      # detect small / distant tags
        self.p = lambda n: self.get_parameter(n).value

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'DetectorParameters'):
            self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        for attr, val in (('cornerRefinementMethod', cv2.aruco.CORNER_REFINE_SUBPIX),
                          ('minMarkerPerimeterRate', float(self.p('min_perimeter_rate')))):
            try:
                setattr(self.aruco_params, attr, val)
            except Exception:
                pass
        self.detector = None
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.K = None
        self.D = np.zeros(5)
        self.odom = None                    # (x, y, yaw)
        self.t_img = None                   # last image time
        self.t_seen = None                  # last detection time
        self.X = self.Z = None              # filtered camera-frame position of the tag
        self.d_f = None
        self.rr = 0.0                       # filtered range rate
        self.last_bearing = 0.0
        self.leader_odom = None             # last known leader position in the odom frame
        self.mode = 'SEARCH'                # SEARCH until the tag has been seen once
        self.mode_t0 = self.now()
        self.search_dir = 1.0
        self.search_accum = 0.0
        self.v_out = 0.0
        self.t_ctl = self.now()
        self.t_log = 0.0
        self.t_warn = 0.0
        self.sub_img = self.sub_info = self.sub_odom = None

        self.cmd_pub = self.create_publisher(Twist, self.p('cmd_topic'), 10)
        self.t_boot = self.now()
        self.create_timer(1.0, self.discover)
        self.create_timer(0.05, self.control)
        self.get_logger().info('Follower node started - waiting for camera / odom topics')

    # ------------------------------------------------------------------ plumbing
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def discover(self):
        need_img = self.sub_img is None
        need_info = self.sub_info is None
        need_odom = self.sub_odom is None
        if not (need_img or need_info or need_odom):
            return
        found = {}
        for name, types in self.get_topic_names_and_types():
            low = name.lower()
            if 'follower' not in low:
                continue
            for t in types:
                if t == 'sensor_msgs/msg/Image' and 'depth' not in low:
                    if 'img' not in found or 'image_raw' in low:
                        found['img'] = name
                elif t == 'sensor_msgs/msg/CameraInfo' and 'depth' not in low:
                    found.setdefault('info', name)
                elif t == 'nav_msgs/msg/Odometry':
                    found.setdefault('odom', name)
        timeout = self.now() - self.t_boot > 15.0
        if need_img:
            topic = self.p('image_topic') or found.get('img') or ('/follower/camera/image_raw' if timeout else '')
            if topic:
                self.sub_img = self.create_subscription(Image, topic, self.image_cb, qos_profile_sensor_data)
                self.get_logger().info(f'Camera image: {topic}')
        if need_info:
            topic = self.p('camera_info_topic') or found.get('info') or ('/follower/camera/camera_info' if timeout else '')
            if topic:
                self.sub_info = self.create_subscription(CameraInfo, topic, self.info_cb, qos_profile_sensor_data)
                self.get_logger().info(f'Camera info: {topic}')
        if need_odom:
            topic = self.p('odom_topic') or found.get('odom') or ('/follower/odom' if timeout else '')
            if topic:
                self.sub_odom = self.create_subscription(Odometry, topic, self.odom_cb, qos_profile_sensor_data)
                self.get_logger().info(f'Odometry: {topic}')
        if timeout and self.t_img is None:
            self.get_logger().warning('No follower camera frames yet. Topics seen: '
                                      f'{[n for n, _ in self.get_topic_names_and_types() if "follower" in n.lower()]}'
                                      ' - set image_topic / camera_info_topic / odom_topic if wrong')

    def info_cb(self, msg):
        if msg.k[0] > 1.0:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            dd = np.array(msg.d, dtype=np.float64)
            self.D = dd if dd.size else np.zeros(5)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    # ------------------------------------------------------------------ vision
    def intrinsics(self, shape):
        if self.K is not None:
            return self.K, self.D
        h, w = shape[:2]
        f = (w / 2.0) / math.tan(float(self.p('camera_hfov')) / 2.0)
        return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]]), np.zeros(5)

    def image_cb(self, msg):
        t = self.now()
        self.t_img = t
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            if ids is None:
                return
            sel = [c for i, c in zip(ids.flatten(), corners) if int(i) == int(self.p('tag_id'))]
            if not sel:
                return
            c = max(sel, key=lambda q: cv2.contourArea(q.reshape(4, 2).astype(np.float32)))
            K, D = self.intrinsics(frame.shape)
            s = float(self.p('tag_size')) / 2.0
            obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2).astype(np.float64), K, D,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                return
            X, _, Z = [float(v) for v in tvec.reshape(3)]
            if Z < 0.05 or math.hypot(X, Z) > 10.0:
                return
            self.update_measurement(X, Z, t)
        except Exception:
            self.get_logger().error('image_cb failed:\n' + traceback.format_exc())

    def update_measurement(self, X, Z, t):
        Z = Z + float(self.p('cam_forward_offset'))
        d = math.hypot(X, Z)
        if self.t_seen is not None and t - self.t_seen < 0.5 and self.d_f is not None:
            dt = max(t - self.t_seen, 1e-3)
            a = 0.5
            self.X = a * X + (1 - a) * self.X
            self.Z = a * Z + (1 - a) * self.Z
            d_new = math.hypot(self.X, self.Z)
            rr = clamp((d_new - self.d_f) / dt, -1.0, 1.0)
            self.rr = 0.7 * self.rr + 0.3 * rr
            self.d_f = d_new
        else:                                # (re)acquired: no history
            self.X, self.Z, self.d_f, self.rr = X, Z, d, 0.0
        self.t_seen = t
        self.last_bearing = math.atan2(-self.X, self.Z)      # + = leader to the left
        if self.odom is not None:
            rx, ry, yaw = self.odom
            fwd, left = self.Z, -self.X
            self.leader_odom = (rx + fwd * math.cos(yaw) - left * math.sin(yaw),
                                ry + fwd * math.sin(yaw) + left * math.cos(yaw))

    # ------------------------------------------------------------------ control laws
    def track_cmd(self, d, b, rr):
        """Tag visible: range d, bearing b (+ = leader on the left), range rate rr."""
        P = self.p
        w = clamp(float(P('kw')) * b, -float(P('w_max')), float(P('w_max')))
        if abs(b) > float(P('turn_in_place')):
            return 0.0, w                                    # face the leader first
        if d < float(P('d_min')) and P('allow_reverse'):
            return -float(P('v_rev')), w                     # leader too close: back off slowly
        e = d - float(P('d_des'))
        band = float(P('d_deadband'))
        v = float(P('kv')) * (e - band) if e > band else 0.0
        if e > -band:
            v += float(P('k_ff')) * max(rr, 0.0)             # leader moving away: keep up
        v = clamp(v, 0.0, float(P('v_max')))
        v *= max(0.0, math.cos(b)) ** 2                      # never full speed while turning hard
        return v, w

    def hold_cmd(self):
        """Tag lost: keep the camera on the leader's last known position; creep to it only if far."""
        v, w = 0.0, 0.0
        if self.odom is not None and self.leader_odom is not None:
            rx, ry, yaw = self.odom
            dx, dy = self.leader_odom[0] - rx, self.leader_odom[1] - ry
            err = wrap(math.atan2(dy, dx) - yaw)
            dist = math.hypot(dx, dy)
            w = clamp(1.5 * err, -0.6, 0.6)
            if dist > float(self.p('d_des')) + 0.3 and abs(err) < 0.3:
                v = min(0.2, 0.5 * (dist - float(self.p('d_des'))))
        return v, w

    def control(self):
        try:
            self._control()
        except Exception:
            self.get_logger().error('control failed:\n' + traceback.format_exc())
            self.publish(0.0, 0.0)

    def _control(self):
        t = self.now()
        dt = clamp(t - self.t_ctl, 1e-3, 0.2)
        self.t_ctl = t
        if self.t_img is None or t - self.t_img > 1.0:
            if t - self.t_warn > 3.0:
                self.t_warn = t
                self.get_logger().warning('No camera frames - holding still')
            self.publish(0.0, 0.0, dt)
            return

        seen = self.t_seen is not None and (t - self.t_seen) < float(self.p('lost_after'))
        if seen:
            if self.mode != 'TRACK':
                self.get_logger().info('Tag 49 acquired - tracking')
            self.mode = 'TRACK'
            v, w = self.track_cmd(self.d_f, self.last_bearing, self.rr)
            if t - self.t_log > 1.0:
                self.t_log = t
                od = f'Odom X={self.odom[0]:.2f} Y={self.odom[1]:.2f}' if self.odom else 'Odom n/a'
                self.get_logger().info(f'Leader ID {int(self.p("tag_id"))} | Cam X={self.X:.2f} Z={self.Z:.2f} '
                                       f'| D={self.d_f:.2f} m | {od} | Cmd V={v:.2f} W={w:.2f}')
        else:
            if self.mode == 'TRACK':
                self.mode, self.mode_t0 = 'HOLD', t
                self.v_out = min(self.v_out, 0.0)            # drop any stale forward command
                self.get_logger().warning('Tag lost - holding and watching the last known position')
            if self.mode == 'HOLD':
                v, w = self.hold_cmd()
                if t - self.mode_t0 > float(self.p('hold_sec')):
                    self.mode, self.search_accum = 'SEARCH', 0.0
                    self.search_dir = 1.0 if self.last_bearing >= 0 else -1.0
                    self.get_logger().warning('Still lost - rotating to search')
            else:                                            # SEARCH: rotate in place only
                v, w = 0.0, self.search_dir * float(self.p('search_w'))
                self.search_accum += abs(w) * dt
                if self.leader_odom is not None and self.search_accum > 2 * math.pi:
                    self.mode, self.mode_t0 = 'HOLD', t
                    self.search_dir *= -1.0
            if t - self.t_log > 2.0:
                self.t_log = t
                self.get_logger().info(f'{self.mode}: tag not visible for {t - (self.t_seen or self.t_boot):.0f} s')
        self.publish(v, w, dt)

    def publish(self, v, w, dt=0.05):
        # speed-up limited gently, slow-down limited generously (never keeps pushing blind)
        up, down = float(self.p('acc_lim')) * dt, float(self.p('dec_lim')) * dt
        dv = v - self.v_out
        self.v_out += clamp(dv, -down, up)
        msg = Twist()
        msg.linear.x = float(self.v_out)
        msg.angular.z = float(clamp(w, -float(self.p('w_max')), float(self.p('w_max'))))
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
