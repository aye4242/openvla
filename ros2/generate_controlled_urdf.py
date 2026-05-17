#!/usr/bin/env python3
"""Generate Panda URDF for Gazebo Classic 11 with ros2_control + baked home pose.

合并方案：
1) string-level 替换 package URI / damping / K
2) 把 Panda 标准 home pose 内嵌到 joint origin（joint=0 时物理上即 home pose）
3) 添加 world → base fixed joint（base 抬高到 BASE_Z）
4) 注入 <ros2_control> 块（position 接口 + initial_value 0）
5) 注入 <gazebo> 插件块（gazebo_ros2_control 加载 yaml）
"""
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

URDF_IN = "/tmp/panda.urdf"
URDF_OUT = "/tmp/panda_controlled.urdf"
CONTROLLERS_YAML = "/home/wh/hb/openvla/ros2/config/panda_controllers.yaml"

# Panda 标准 ready / home pose（非零关节）
HOME_ANGLES = {
    "fer_joint2": -0.7854,
    "fer_joint4": -2.3562,
    "fer_joint6":  1.5708,
    "fer_joint7":  0.7854,
}

# ros2_control 暴露的 9 个 joint
JOINTS = [f"fer_joint{i}" for i in range(1, 8)] + ["fer_finger_joint1", "fer_finger_joint2"]

# base 在 world 中的高度（站台等效，让 home pose 末端落在桌面上方）
BASE_Z = 0.4

# --- 1) 字符串级替换 ---
with open(URDF_IN) as f:
    urdf = f.read()

urdf = urdf.replace("package://franka_description", "/opt/ros/humble/share/franka_description")
urdf = urdf.replace('damping="0.003"', 'damping="1.0"')   # ros2_control 主控制，无需大 damping
urdf = urdf.replace('K="7000"', 'K="0"')

# --- 2) ElementTree 解析 + home pose baking ---
root = ET.fromstring(urdf)

for joint in root.iter("joint"):
    name = joint.get("name")
    home = HOME_ANGLES.get(name, 0.0)
    if home == 0.0:
        continue

    # ① origin rpy: R_new = R_orig × R_z(home)
    origin = joint.find("origin")
    if origin is not None:
        r, p, y = map(float, origin.get("rpy", "0 0 0").split())
        R_orig = Rotation.from_euler("xyz", [r, p, y])
        R_new = R_orig * Rotation.from_euler("z", home)
        r_new, p_new, y_new = R_new.as_euler("xyz")
        origin.set("rpy", f"{r_new:.6f} {p_new:.6f} {y_new:.6f}")

    # ② limits 整体平移（关节空间等价）
    limit = joint.find("limit")
    if limit is not None:
        lower = float(limit.get("lower", 0)) - home
        upper = float(limit.get("upper", 0)) - home
        limit.set("lower", f"{lower:.6f}")
        limit.set("upper", f"{upper:.6f}")

    sc = joint.find("safety_controller")
    if sc is not None:
        if sc.get("soft_lower_limit") is not None:
            sc.set("soft_lower_limit", f"{float(sc.get('soft_lower_limit')) - home:.6f}")
        if sc.get("soft_upper_limit") is not None:
            sc.set("soft_upper_limit", f"{float(sc.get('soft_upper_limit')) - home:.6f}")

    print(f"  baked {name}: home={home:+.4f} rad")

# --- 3) 加 fixed world → base joint（base 抬高，防止穿地） ---
world_link = ET.Element("link", {"name": "world"})
world_joint = ET.Element("joint", {"name": "world_to_base", "type": "fixed"})
ET.SubElement(world_joint, "parent", {"link": "world"})
ET.SubElement(world_joint, "child", {"link": "base"})
ET.SubElement(world_joint, "origin", {"rpy": "0 0 0", "xyz": f"0 0 {BASE_Z}"})
root.insert(0, world_link)
root.insert(1, world_joint)

# --- 4) 注入 <ros2_control> 块 ---
rc = ET.SubElement(root, "ros2_control", {"name": "GazeboSystem", "type": "system"})
hardware = ET.SubElement(rc, "hardware")
hw_plugin = ET.SubElement(hardware, "plugin")
hw_plugin.text = "gazebo_ros2_control/GazeboSystem"
for jn in JOINTS:
    j = ET.SubElement(rc, "joint", {"name": jn})
    ET.SubElement(j, "command_interface", {"name": "position"})
    state_pos = ET.SubElement(j, "state_interface", {"name": "position"})
    init_param = ET.SubElement(state_pos, "param", {"name": "initial_value"})
    init_param.text = "0.0"
    ET.SubElement(j, "state_interface", {"name": "velocity"})

# --- 5) 注入 <gazebo> 插件块 ---
gz = ET.SubElement(root, "gazebo")
gz_plugin = ET.SubElement(gz, "plugin", {
    "filename": "libgazebo_ros2_control.so",
    "name": "gazebo_ros2_control",
})
params = ET.SubElement(gz_plugin, "parameters")
params.text = CONTROLLERS_YAML

# --- 6) 写出 ---
ET.ElementTree(root).write(URDF_OUT, encoding="unicode", xml_declaration=False)

print(f"\nGenerated: {URDF_OUT}")
print(f"  - damping 0.003 -> 1.0")
print(f"  - K 7000 -> 0")
print(f"  - home pose baked into joint origins")
print(f"  - world fixed joint @ z={BASE_Z}m")
print(f"  - ros2_control + gazebo plugin injected ({len(JOINTS)} joints)")
