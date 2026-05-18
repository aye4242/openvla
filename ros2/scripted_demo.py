#!/usr/bin/env python3
"""脚本驱动 demo v3：前推时同步下降，解决"高了夹空气"问题。

几何分析（world 坐标）：
- home pose flange z = base_z(0.4) + FK_z(0.59) = 0.99m
- finger tip ≈ flange - 0.107m = 0.88m
- 红块顶部 = 0.775 + 0.025 = 0.80m
- 夹取时 finger tip 应在红块上方 ~2cm = 0.82m
- 所以从 home 到夹取点，finger tip 需下降 0.88-0.82 = 0.06m = 60mm

脚本策略：
- phase1: 前推 280mm + 同步下降 60mm（一步到位到红块上方）
- phase2: 微调下降 20mm（到夹取高度）
- phase3: 闭爪
"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

RATE_HZ = 10.0

DEMO_SEQUENCE = [
    # 1. 前推 280mm + 同步下降 60mm
    #    3.5s×10×0.27×0.03 = 283.5mm x
    #    3.5s×10×0.06×0.03 = 63mm z 下降
    ("1. approach → 红块上方",  3.5, [+0.30, 0.00, -0.06, 0, 0, 0], 0.0),

    # 2. 微调下降 20mm（到夹取高度，finger tip 在红块上方 2cm）
    ("2. fine descent",          1.0, [ 0.00, 0.00, -0.07, 0, 0, 0], 0.0),

    # 3. 半闭夹取（gripper=0.5 → finger=0.02，刚好夹住不挤压弹飞）
    ("3. close gripper",        2.0, [ 0.00, 0.00,  0.00, 0, 0, 0], 0.5),

    # 4. 抬起 180mm（提走红块）
    ("4. lift up",              2.0, [ 0.00, 0.00, +0.30, 0, 0, 0], 1.0),

    # 5. 侧移 120mm
    ("5. side translate",       2.0, [ 0.00, +0.20, 0.00, 0, 0, 0], 1.0),

    # 6. 松爪
    ("6. release",              1.0, [ 0.00, 0.00,  0.00, 0, 0, 0], 0.0),

    # 7. 后退 200mm 归位
    ("7. retreat",              2.0, [-0.33, 0.00, 0.00, 0, 0, 0], 0.0),
]


class DemoDriver(Node):
    def __init__(self):
        super().__init__("scripted_demo")
        self.pub = self.create_publisher(Float64MultiArray, "/vla/action", 10)
        self.get_logger().info("Scripted demo v3 starting in 3s ...")
        time.sleep(3.0)
        self.run_sequence()

    def run_sequence(self):
        dt = 1.0 / RATE_HZ
        for name, secs, delta6, grip in DEMO_SEQUENCE:
            self.get_logger().info(f"▶  {name}  ({secs:.1f}s)")
            steps = int(round(secs * RATE_HZ))
            for _ in range(steps):
                msg = Float64MultiArray()
                msg.data = [float(v) for v in list(delta6) + [float(grip)]]
                self.pub.publish(msg)
                time.sleep(dt)
        self.get_logger().info("Demo done.")


def main():
    rclpy.init()
    node = DemoDriver()
    rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
