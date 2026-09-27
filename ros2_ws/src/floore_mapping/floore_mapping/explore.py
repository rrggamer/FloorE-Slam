#!/usr/bin/env python3
"""Autonomous reactive wander: drives straight until the lidar sees an
obstacle ahead, then turns in place toward whichever side has more open
space, so record_data.py can collect map coverage without a human driving
teleop. This is plain obstacle-avoidance, not SLAM/frontier exploration --
enough to cover a floor plan, which the assignment allows as an
alternative to manual driving.

Reads /scan for obstacle distances and /ground_truth (see
floore.gazebo.xacro's OdometryPublisher) to detect when the robot is stuck
(e.g. wedged against a corner) and needs to back out.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class Explorer(Node):
    def __init__(self):
        super().__init__("explorer")
        self.declare_parameter("linear_speed", 0.4)
        self.declare_parameter("angular_speed", 0.8)
        self.declare_parameter("safe_distance", 1.0)
        self.declare_parameter("front_half_angle_deg", 35.0)
        self.declare_parameter("stuck_timeout", 6.0)
        self.declare_parameter("run_time", 0.0)  # seconds; <= 0 means run until Ctrl+C

        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed = self.get_parameter("angular_speed").value
        self.safe_distance = self.get_parameter("safe_distance").value
        self.front_half_angle = math.radians(self.get_parameter("front_half_angle_deg").value)
        self.stuck_timeout = self.get_parameter("stuck_timeout").value
        run_time = self.get_parameter("run_time").value

        self._scan = None
        self._turn_bias = 1.0  # +1 = turn left, -1 = turn right; flips when stuck
        self._last_pose = None
        self._last_move_time = self.get_clock().now()

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(Odometry, "/ground_truth", self._on_odom, 10)
        self._control_timer = self.create_timer(0.1, self._control_loop)
        if run_time > 0:
            self.create_timer(run_time, self._time_up)

        self.get_logger().info(
            "Explorer started: wandering" + (" until Ctrl+C" if run_time <= 0 else f" for {run_time:.0f}s")
        )

    def _on_scan(self, msg):
        self._scan = msg

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        if self._last_pose is not None:
            moved = math.hypot(p.x - self._last_pose[0], p.y - self._last_pose[1])
            if moved > 0.03:
                self._last_move_time = self.get_clock().now()
        self._last_pose = (p.x, p.y)

    def _time_up(self):
        self._control_timer.cancel()
        self._cmd_pub.publish(Twist())
        self.get_logger().info("run_time reached, stopping")
        rclpy.shutdown()

    def _control_loop(self):
        if self._scan is None:
            return

        ranges = self._scan.ranges
        n = len(ranges)
        angle_min = self._scan.angle_min
        angle_inc = self._scan.angle_increment

        def idx(angle):
            i = int(round((angle - angle_min) / angle_inc))
            return max(0, min(n - 1, i))

        front_lo, front_hi = idx(-self.front_half_angle), idx(self.front_half_angle)
        front = [r for r in ranges[front_lo:front_hi + 1] if math.isfinite(r) and r > 0]
        front_clear = min(front) if front else self._scan.range_max

        stuck = (self.get_clock().now() - self._last_move_time).nanoseconds / 1e9 > self.stuck_timeout

        cmd = Twist()
        if stuck:
            # Spinning near an obstacle without making progress (e.g. wedged
            # in a corner) -- flip turn direction and back up to break the
            # deadlock.
            self._turn_bias *= -1.0
            cmd.linear.x = -0.15
            cmd.angular.z = self.angular_speed * self._turn_bias
        elif front_clear < self.safe_distance:
            left = [r for r in ranges[idx(0.0):idx(math.pi / 2) + 1] if math.isfinite(r) and r > 0]
            right = [r for r in ranges[idx(-math.pi / 2):idx(0.0) + 1] if math.isfinite(r) and r > 0]
            left_clear = sum(left) / len(left) if left else 0.0
            right_clear = sum(right) / len(right) if right else 0.0
            self._turn_bias = 1.0 if left_clear >= right_clear else -1.0
            cmd.angular.z = self.angular_speed * self._turn_bias
        else:
            cmd.linear.x = self.linear_speed

        self._cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = Explorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
