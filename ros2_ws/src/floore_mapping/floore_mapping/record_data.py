#!/usr/bin/env python3
"""Record synchronized (odom pose, lidar scan) pairs to a CSV file, in the
same row layout as the CoppeliaSim vrep-OCGM assignment's
save_laser_show_pointcloud.py:

    col 0        = x position of the sensor base (odom frame)
    col 1        = y position of the sensor base (odom frame)
    col 2        = theta, sensor base heading about z (radians)
    col 3+2*i    = range of ray i (meters)
    col 4+2*i    = angle of ray i relative to the sensor's forward axis (radians)

/odom (30 Hz) and /scan (10 Hz) are bridged from Gazebo by floore_gazebo's
launch file, so this only needs to run alongside it -- see
floore_mapping/launch/record.launch.py.
"""
import csv
import math

import message_filters
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ScanRecorder(Node):
    def __init__(self):
        super().__init__("scan_recorder")
        self.declare_parameter("out_file", "data.csv")
        out_path = self.get_parameter("out_file").get_parameter_value().string_value

        self._file = open(out_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._count = 0

        odom_sub = message_filters.Subscriber(self, Odometry, "/odom")
        scan_sub = message_filters.Subscriber(self, LaserScan, "/scan")
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [odom_sub, scan_sub], queue_size=30, slop=0.1
        )
        self._sync.registerCallback(self._callback)

        self.get_logger().info(f"Recording synchronized /odom + /scan to {out_path}")

    def _callback(self, odom, scan):
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        th = yaw_from_quaternion(odom.pose.pose.orientation)

        row = [x, y, th]
        for i, r in enumerate(scan.ranges):
            # gz-sim reports out-of-range rays as +inf; treat them as a
            # max-range (no obstacle detected) reading like a real Hokuyo.
            if not math.isfinite(r):
                r = scan.range_max
            angle = scan.angle_min + i * scan.angle_increment
            row.extend([r, angle])

        self._writer.writerow(row)
        self._count += 1
        if self._count % 20 == 0:
            self._file.flush()
            self.get_logger().info(f"{self._count} scans recorded")

    def destroy_node(self):
        self._file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = ScanRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
