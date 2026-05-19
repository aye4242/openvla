#!/usr/bin/env python3
"""FK 校验：q_controlled=[0]*7 时，KDL 算出的位姿 vs Gazebo 实际位姿。"""
import math
import numpy as np
import PyKDL as kdl

HOME_OFFSET = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])


def build_panda_chain() -> kdl.Chain:
    chain = kdl.Chain()
    mdh = [
        (0.0,      0.0,         0.333),
        (0.0,     -math.pi / 2, 0.0),
        (0.0,      math.pi / 2, 0.316),
        (0.0825,   math.pi / 2, 0.0),
        (-0.0825, -math.pi / 2, 0.384),
        (0.0,      math.pi / 2, 0.0),
        (0.088,    math.pi / 2, 0.0),
    ]
    for i, (a, alpha, d) in enumerate(mdh):
        if not (i == 0 and a == 0.0 and alpha == 0.0):
            pre = (kdl.Frame(kdl.Rotation.RotX(alpha))
                   * kdl.Frame(kdl.Vector(a, 0.0, 0.0)))
            chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.Fixed), pre))
        chain.addSegment(kdl.Segment(
            kdl.Joint(kdl.Joint.RotZ),
            kdl.Frame(kdl.Vector(0.0, 0.0, d)),
        ))
    chain.addSegment(kdl.Segment(
        kdl.Joint(kdl.Joint.Fixed),
        kdl.Frame(kdl.Vector(0.0, 0.0, 0.107)),
    ))
    return chain


def kdl_to_array(frame: kdl.Frame):
    """KDL Frame → (x,y,z, r,p,y)"""
    p = frame.p
    rpy = frame.M.GetRPY()
    return np.array([p.x(), p.y(), p.z(), rpy[0], rpy[1], rpy[2]])


chain = build_panda_chain()
fk = kdl.ChainFkSolverPos_recursive(chain)

q_ctrl = np.zeros(7)  # 控制零位 = 物理 ready pose
q_kdl = q_ctrl + HOME_OFFSET

q_arr = kdl.JntArray(7)
for i in range(7):
    q_arr[i] = float(q_kdl[i])

ee_frame = kdl.Frame()
fk.JntToCart(q_arr, ee_frame)

pose = kdl_to_array(ee_frame)
print(f"q_controlled = {q_ctrl}")
print(f"q_kdl        = {q_kdl}")
print(f"KDL FK EEF pose (x,y,z, r,p,y):")
print(f"  pos:  [{pose[0]:+.4f}, {pose[1]:+.4f}, {pose[2]:+.4f}]")
print(f"  rot:  [{pose[3]:+.4f}, {pose[4]:+.4f}, {pose[5]:+.4f}]")

# 也请主人在另一个终端运行：
# ros2 topic echo /tf --once  # 找 fer_hand 或 fer_link8 的位姿
# 或者：gz model -m panda -p  # 看末端位置
print("\n请对比 Gazebo 中 fer_hand/fer_link8 的实际位姿！")
