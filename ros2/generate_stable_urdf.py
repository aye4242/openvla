#!/usr/bin/env python3
"""Generate Panda URDF for Gazebo Classic 11.

3 步处理：
1. 替换 package:// URI 为绝对路径
2. damping 0.003 → 5.0（适度阻尼）
3. 把 home pose 角度"内嵌"到 joint origin（数学上 R_origin' = R_origin · R_z(home_angle)），
   使 joint=0 时机械臂物理上处于 home pose 姿态。
   这绕开了 Gazebo Classic 11 不读取 SDF <initial_position> 的问题。
"""
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

URDF_IN = "/tmp/panda.urdf"
URDF_OUT = "/tmp/panda_stable.urdf"

# Panda home pose 中非零的 joint
HOME_ANGLES = {
    "fer_joint2": -0.7854,
    "fer_joint4": -2.3562,
    "fer_joint6":  1.5708,
    "fer_joint7":  0.7854,
}

# 1. 读取并 string-level 替换
with open(URDF_IN) as f:
    urdf = f.read()

urdf = urdf.replace("package://franka_description", "/opt/ros/humble/share/franka_description")
urdf = urdf.replace('damping="0.003"', 'damping="50.0"')
urdf = urdf.replace('K="7000"', 'K="0"')

# 2. ElementTree 平移 home pose 到 joint origin
root = ET.fromstring(urdf)

for joint in root.iter("joint"):
    name = joint.get("name")
    home = HOME_ANGLES.get(name, 0.0)
    if home == 0.0:
        continue

    # ① origin: R_new = R_origin · R_z(home)
    origin = joint.find("origin")
    if origin is not None:
        r, p, y = map(float, origin.get("rpy", "0 0 0").split())
        R_orig = Rotation.from_euler("xyz", [r, p, y])
        R_new = R_orig * Rotation.from_euler("z", home)
        r_new, p_new, y_new = R_new.as_euler("xyz")
        origin.set("rpy", f"{r_new:.6f} {p_new:.6f} {y_new:.6f}")

    # ② limits: 整体平移（关节空间等价）
    limit = joint.find("limit")
    if limit is not None:
        lower = float(limit.get("lower", 0)) - home
        upper = float(limit.get("upper", 0)) - home
        limit.set("lower", f"{lower:.6f}")
        limit.set("upper", f"{upper:.6f}")
        # 同步 safety_controller 的 soft limit（如有）
    sc = joint.find("safety_controller")
    if sc is not None:
        if sc.get("soft_lower_limit") is not None:
            sc.set("soft_lower_limit", f"{float(sc.get('soft_lower_limit')) - home:.6f}")
        if sc.get("soft_upper_limit") is not None:
            sc.set("soft_upper_limit", f"{float(sc.get('soft_upper_limit')) - home:.6f}")

    print(f"  baked {name}: home={home:.4f}  rpy=({r_new:.4f},{p_new:.4f},{y_new:.4f})")

# 3. 把 base 固定到 world（防止整个模型因重力下沉穿地）
world_link = ET.Element("link", {"name": "world"})
world_joint = ET.Element("joint", {"name": "world_to_base", "type": "fixed"})
ET.SubElement(world_joint, "parent", {"link": "world"})
ET.SubElement(world_joint, "child", {"link": "base"})
ET.SubElement(world_joint, "origin", {"rpy": "0 0 0", "xyz": "0 0 0.8"})
root.insert(0, world_link)
root.insert(1, world_joint)

# 4. Add joint state publisher plugin for ROS2 feedback
gazebo_elem = ET.SubElement(root, "gazebo")
plugin = ET.SubElement(gazebo_elem, "plugin")
plugin.set("name", "joint_state_publisher")
plugin.set("filename", "libgazebo_ros_joint_state_publisher.so")
ros_elem = ET.SubElement(plugin, "ros")
ns_elem = ET.SubElement(ros_elem, "namespace")
ns_elem.text = "/"
rate = ET.SubElement(plugin, "update_rate")
rate.text = "20"
# Explicit joint names (required by gazebo_ros2)
ALL_JOINTS = [f"fer_joint{i}" for i in range(1, 8)] + ["fer_finger_joint1", "fer_finger_joint2"]
for jname in ALL_JOINTS:
    jn = ET.SubElement(plugin, "joint_name")
    jn.text = jname

# 5. Write
ET.ElementTree(root).write(URDF_OUT, encoding="unicode", xml_declaration=False)

print(f"Generated: {URDF_OUT}")
print("Changes: damping 0.003 → 50.0, home pose baked, joint_state_publisher plugin added")
