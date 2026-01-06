# Franka 数据集 State 和 Action 设置说明

## 概述

数据集的 state 和 action 通过三个步骤设置：

1. **数据提取**：从 ROS2 bag 文件中提取原始数据
2. **数据存储**：将提取的数据存储到 HDF5 文件
3. **数据组合**：从 HDF5 读取并组合为最终的 state/action 向量

## 第一步：数据提取（processors_franka.py）

从 ROS2 bag 文件中提取数据，每个 processor 负责处理特定类型的消息：

### 1. 关节状态（JointState）
```python
# processors_franka.py: FrankaJointStateProcessor
# 处理 /left/franka/joint_states_desired 和 /right/franka/joint_states_desired

data['state'][f'{arm_side}_franka/joint_positions'] = np.array(joint_positions[:7])
data['action'][f'{arm_side}_franka/joint_positions'] = np.array(joint_positions[:7])
```

**说明**：
- 提取前 7 个关节位置（Franka 有 7 个自由度）
- **State 和 Action 使用相同的数据**（因为这是 desired joint states，表示期望的关节位置）

### 2. 末端执行器位姿（PoseStamped）
```python
# processors_franka.py: FrankaEEFPoseProcessor
# 处理 /left/franka/eef_pose 和 /right/franka/eef_pose

pose_array = [x, y, z, qx, qy, qz, qw]  # 7维：位置 + 四元数
data['state'][f'{arm_side}_franka/eef_pose'] = pose_array
data['action'][f'{arm_side}_franka/eef_pose'] = pose_array
```

**说明**：
- 提取位置（x, y, z）和四元数（qx, qy, qz, qw）
- **State 和 Action 使用相同的数据**（当前位姿）

### 3. 夹爪状态（JointState）
```python
# processors_franka.py: FrankaGripperStateProcessor
# 处理 /left/franka/gripper_state 和 /right/franka/gripper_state

gripper_position = msg.position[0]  # 第一个关节位置
data['state'][f'{arm_side}_franka/gripper_position'] = np.array([gripper_position])
data['action'][f'{arm_side}_franka/gripper_position'] = np.array([gripper_position])
```

**说明**：
- 提取夹爪的第一个关节位置
- **State 和 Action 使用相同的数据**

## 第二步：数据存储（ros2_to_lerobot_converter.py）

将提取的数据存储到 HDF5 文件，结构如下：

```
data.h5
├── state/
│   ├── left_franka_joint_states/
│   │   └── left_franka/joint_positions  (dataset: [N, 7])
│   ├── right_franka_joint_states/
│   │   └── right_franka/joint_positions  (dataset: [N, 7])
│   ├── left_franka_eef/
│   │   └── left_franka/eef_pose  (dataset: [N, 7])
│   ├── right_franka_eef/
│   │   └── right_franka/eef_pose  (dataset: [N, 7])
│   ├── left_franka_gripper/
│   │   └── left_franka/gripper_position  (dataset: [N, 1])
│   └── right_franka_gripper/
│       └── right_franka/gripper_position  (dataset: [N, 1])
└── action/
    └── (相同的结构)
```

**注意**：
- 如果某个字段只有 state 没有 action，会自动生成 action（使用下一帧的 state）
- 参考代码：`ros2_to_lerobot_converter.py:375-380`

## 第三步：数据组合（custom_state_action_mapping_franka.py）

从 HDF5 读取数据并组合为最终的 state/action 向量：

### State 组合（28 维）

```python
def combine_franka_dual_arm_state(components):
    state_parts = []
    
    # 1. 左臂关节（7维）
    state_parts.append(left_joints[:7])
    
    # 2. 右臂关节（7维）
    state_parts.append(right_joints[:7])
    
    # 3. 左末端执行器位姿（6维：x, y, z, roll, pitch, yaw）
    state_parts.append(pose_to_6d(left_ee_pose))  # 7维转6维
    
    # 4. 右末端执行器位姿（6维）
    state_parts.append(pose_to_6d(right_ee_pose))
    
    # 5. 左夹爪位置（1维）
    state_parts.append(left_gripper)
    
    # 6. 右夹爪位置（1维）
    state_parts.append(right_gripper)
    
    return np.concatenate(state_parts)  # 28维
```

**State 向量结构**：
```
[左臂关节(7) | 右臂关节(7) | 左EEF(6) | 右EEF(6) | 左夹爪(1) | 右夹爪(1)]
= 7 + 7 + 6 + 6 + 1 + 1 = 28 维
```

### Action 组合（28 维）

```python
def combine_franka_dual_arm_action(components):
    # Action 使用相同的组合逻辑
    # 因为数据采集时，action = 当前状态（desired states）
    return combine_franka_dual_arm_state(components)
```

**Action 向量结构**：
```
[左臂关节命令(7) | 右臂关节命令(7) | 左EEF命令(6) | 右EEF命令(6) | 左夹爪命令(1) | 右夹爪命令(1)]
= 7 + 7 + 6 + 6 + 1 + 1 = 28 维
```

## 关键点说明

### 1. 为什么 State 和 Action 使用相同的数据？

**原因**：
- 数据来自 `/left/franka/joint_states_desired`，这是**期望的关节状态**，不是实际状态
- 在演示数据中，期望状态 = 当前命令 = 下一帧的状态
- 因此，`action[t] = state[t+1]`（近似）

**实际应用**：
- 训练时，模型学习：`action = policy(state)`
- 推理时，模型预测：给定当前 state，输出 action

### 2. 位姿转换（7维 → 6维）

```python
def pose_to_6d(pose):
    # 输入：7维 [x, y, z, qx, qy, qz, qw]
    # 输出：6维 [x, y, z, roll, pitch, yaw]
    
    position = pose[:3]  # x, y, z
    quaternion = pose[3:7]  # qx, qy, qz, qw
    
    # 四元数转欧拉角
    roll, pitch, yaw = quaternion_to_euler(quaternion)
    
    return [x, y, z, roll, pitch, yaw]
```

**为什么转换**：
- 四元数（4维）有约束条件，不适合直接用于神经网络
- 欧拉角（3维）更直观，但可能有万向锁问题
- 6维表示（位置+欧拉角）是常用的折中方案

### 3. 数据路径映射

**HDF5 路径** → **Mapping 文件路径**：
```
state/left_franka_joint_states/left_franka/joint_positions
→ "left_franka_joint_states/left_franka/joint_positions"
```

**提取流程**：
```python
# synced_to_lerobot_converter.py:643-648
state_components = extract_components(f["state"], frame_idx, mapping.state_components)
action_components = extract_components(f["action"], frame_idx, mapping.action_components)

state = mapping.state_combine_fn(state_components)  # 组合为28维
action = mapping.action_combine_fn(action_components)  # 组合为28维
```

## 修改 State/Action 的方法

如果需要修改 state 或 action 的结构，编辑 `custom_state_action_mapping_franka.py`：

### 示例1：只使用关节位置（14维）

```python
def combine_franka_dual_arm_state(components):
    state_parts = []
    state_parts.append(left_joints[:7])
    state_parts.append(right_joints[:7])
    return np.concatenate(state_parts)  # 14维
```

### 示例2：添加关节速度

```python
state_components = [
    "left_franka_joint_states/left_franka/joint_positions",
    "left_franka_joint_states/left_franka/joint_velocities",  # 新增
    # ...
]

def combine_franka_dual_arm_state(components):
    state_parts = []
    state_parts.append(left_joints[:7])
    state_parts.append(left_velocities[:7])  # 新增
    # ...
    return np.concatenate(state_parts)  # 35维 (7+7+7+7+6+6+1+1)
```

### 示例3：使用下一帧作为 Action

```python
def combine_franka_dual_arm_action(components, next_frame_components):
    # 使用下一帧的状态作为 action
    return combine_franka_dual_arm_state(next_frame_components)
```

## 总结

- **State**：当前时刻的机器人状态（关节、位姿、夹爪）
- **Action**：当前时刻的命令（与 state 相同，因为是 desired states）
- **维度**：28 维（7+7+6+6+1+1）
- **数据来源**：ROS2 bag 文件中的 `/left/franka/joint_states_desired` 等话题

