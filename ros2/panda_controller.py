#!/usr/bin/env python3
"""Panda 位置控制器（ros2_control + 雅可比伪逆 IK）。

订阅 /vla/action（7D = 6D EEF delta + 1D gripper）
  → 用阻尼最小二乘 IK 把 6D EEF delta 映射成 7-DOF joint delta
  → 累加到目标位置后发布到 /forward_position_controller/commands（9D 绝对位置）。

变更（vs 方案 B 简化版）：
- 6D EEF delta 不再直接当 joint delta（语义错误，是之前机械臂只会抬手不去够物体的根因）
- 在线计算雅可比 J(q)，q_dot = J^T (J J^T + λ²I)^-1 × delta_eef
- 阻尼系数 λ=0.05 避免奇异点附近爆炸
- 运动学链用 PyKDL 手工硬编码 Panda 标准 DH 参数（零额外依赖）
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import PyKDL as kdl

# joint 名顺序必须严格与 controllers.yaml 中 `joints:` 列表一致
ARM_JOINTS = [f"fer_joint{i}" for i in range(1, 8)]
GRIPPER_JOINTS = ["fer_finger_joint1", "fer_finger_joint2"]
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS  # 7 + 2 = 9

# Home pose（已 baked 到 URDF，所以控制层全 0）
HOME = np.zeros(9)

# baked 后的关节限位（对应 /tmp/panda_controlled.urdf）
JOINT_LOWER = np.array([-2.8973, -0.9774, -2.8973, -0.7156, -2.8973, -1.5883, -3.6827, 0.0, 0.0])
JOINT_UPPER = np.array([ 2.8973,  2.5482,  2.8973,  2.2864,  2.8973,  2.1817,  2.1119, 0.04, 0.04])

# URDF baked 的 home 角度（Panda ready pose 中非零的关节）
# 物理几何上 q_kdl = q_controlled + HOME_OFFSET（baking 数学：origin × R_z(home) × R_z(q)）
HOME_OFFSET = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])

DEFAULT_SCALE = 0.03      # VLA 6D delta → EEF 实际位移缩放系数（米 / 弧度）
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.0
GRIPPER_STEP = 0.002      # 每帧 finger 最大位移（20Hz → 40mm/s，防弹飞）
PUBLISH_RATE = 20.0
DLS_LAMBDA = 0.05         # 阻尼最小二乘正则化（避免奇异点处 J^-1 爆炸）


def build_panda_chain() -> kdl.Chain:
    """构造 Franka Panda 7-DOF KDL 链 + flange 末端段（Modified DH / Craig）。

    踩坑记录：PyKDL 的 `Frame.DH_Craig1989(a, α, d, θ) + Joint.RotZ`
    实际计算的是 `R_z(q) × frame`，但 Craig 标准要求 `R_x(α) × D_x(a) × R_z(q) × D_z(d)`
    —— PyKDL 把 R_z(q) 放在了最前面，与 Craig 约定顺序颠倒。

    正确做法：把每个关节拆成 [fixed pre-frame: R_x(α) × D_x(a)] +
    [RotZ joint] + [fixed post-frame: D_z(d)] 三段，显式控制变换顺序。

    Joint 总数仍为 7（fixed segment 不算 active joint）。
    DH 参数来自 Franka 官方控制文档。
    """
    chain = kdl.Chain()
    # (a_{i-1}, α_{i-1}, d_i) for i=1..7  —— Craig Modified DH 约定
    mdh = [
        (0.0,      0.0,         0.333),  # joint 1
        (0.0,     -math.pi / 2, 0.0),    # joint 2
        (0.0,      math.pi / 2, 0.316),  # joint 3
        (0.0825,   math.pi / 2, 0.0),    # joint 4
        (-0.0825, -math.pi / 2, 0.384),  # joint 5
        (0.0,      math.pi / 2, 0.0),    # joint 6
        (0.088,    math.pi / 2, 0.0),    # joint 7
    ]
    for i, (a, alpha, d) in enumerate(mdh):
        # joint 1 的 pre-frame 是 identity，可省略 fixed segment
        if not (i == 0 and a == 0.0 and alpha == 0.0):
            pre = (kdl.Frame(kdl.Rotation.RotX(alpha))
                   * kdl.Frame(kdl.Vector(a, 0.0, 0.0)))
            chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.Fixed), pre))
        chain.addSegment(kdl.Segment(
            kdl.Joint(kdl.Joint.RotZ),
            kdl.Frame(kdl.Vector(0.0, 0.0, d)),
        ))
    # 法兰盘 → EEF 偏移 0.107m（Franka 标准）
    chain.addSegment(kdl.Segment(
        kdl.Joint(kdl.Joint.Fixed),
        kdl.Frame(kdl.Vector(0.0, 0.0, 0.107)),
    ))
    return chain


class PandaController(Node):
    def __init__(self):
        super().__init__("panda_controller")
        self.declare_parameter("action_scale", DEFAULT_SCALE)

        self.target_positions = HOME.copy()

        # KDL: chain + FK + Jacobian solver（缓冲区复用，避免每帧分配）
        self.chain = build_panda_chain()
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.q_kdl = kdl.JntArray(7)
        self.J_kdl = kdl.Jacobian(7)
        self.ee_frame_kdl = kdl.Frame()

        self.pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )
        self.create_subscription(
            Float64MultiArray, "/vla/action", self.action_callback, 10
        )
        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.publish_command)

        scale = self.get_parameter("action_scale").value
        self.get_logger().info(
            f"Panda controller ready (KDL Jacobian IK, scale={scale}, rate={PUBLISH_RATE}Hz). "
            f"Locking home pose [0]*9 ..."
        )

    def _compute_jacobian(self, q_phys_7: np.ndarray) -> np.ndarray:
        """根据当前 7 关节"控制层"角度计算雅可比矩阵 (6x7)。"""
        q_kdl_arr = q_phys_7 + HOME_OFFSET  # 控制零位 → KDL 标准零位
        for i in range(7):
            self.q_kdl[i] = float(q_kdl_arr[i])
        self.jac_solver.JntToJac(self.q_kdl, self.J_kdl)
        J = np.zeros((6, 7))
        for r in range(6):
            for c in range(7):
                J[r, c] = self.J_kdl[r, c]
        return J

    def action_callback(self, msg: Float64MultiArray):
        action = np.array(msg.data, dtype=float)
        if action.size != 7:
            self.get_logger().warn(f"Expected 7D action, got {action.size}D, ignored")
            return

        scale = self.get_parameter("action_scale").value

        # VLA 原始输出日志（供分析 sim2real gap）
        self.get_logger().info(
            f"VLA raw: dx={action[0]:+.4f}, dy={action[1]:+.4f}, dz={action[2]:+.4f}, "
            f"dr={action[3]:+.4f}, dp={action[4]:+.4f}, dy={action[5]:+.4f}, grip={action[6]:.2f}"
        )

        # 6D EEF delta（前 3 维 m/step 平移，后 3 维 rad/step 旋转）
        delta_eef = action[:6] * scale  # (6,)

        # 方案 A：屏蔽 dz，只保留水平面 dx+dy（避免 VLA 在 Gazebo 中过度下压）
        delta_eef[2] = 0.0

        # 当前位姿处的雅可比 6×7
        J = self._compute_jacobian(self.target_positions[:7])

        # 阻尼最小二乘 IK: delta_q = J^T (J J^T + λ²I)^-1 × delta_eef
        JJt = J @ J.T + (DLS_LAMBDA ** 2) * np.eye(6)
        delta_q = J.T @ np.linalg.solve(JJt, delta_eef)  # (7,)

        # 累加到 7 关节
        self.target_positions[:7] += delta_q

        # gripper：比例控制（0=全开 0.04, 0.5=半闭 0.02, 1.0=全闭 0.0）
        # 平滑过渡：每帧最多移动 GRIPPER_STEP，避免 Gazebo 高速碰撞弹飞物体
        gripper_cmd = GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN) * float(np.clip(action[6], 0.0, 1.0))
        for i in (7, 8):
            diff = gripper_cmd - self.target_positions[i]
            if abs(diff) > GRIPPER_STEP:
                self.target_positions[i] += float(np.sign(diff)) * GRIPPER_STEP
            else:
                self.target_positions[i] = gripper_cmd

        # clip 到关节限位（防溢出）
        np.clip(self.target_positions, JOINT_LOWER, JOINT_UPPER, out=self.target_positions)

    def publish_command(self):
        msg = Float64MultiArray()
        msg.data = self.target_positions.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PandaController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
