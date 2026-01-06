#!/usr/bin/env python3
"""
Custom State-Action Mapping for Franka Dual-Arm Robot

This file defines how to map HDF5 data to LeRobot state and action tensors
for Franka dual-arm robot configuration.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field
from typing import Callable, Optional
import h5py


@dataclass
class StateActionMapping:
    """Define how to map HDF5 data to LeRobot state and action tensors."""
    
    # State components to combine
    state_components: List[str] = field(default_factory=list)
    
    # Action components to combine  
    action_components: List[str] = field(default_factory=list)
    
    # Custom combine functions
    state_combine_fn: Optional[Callable] = None
    action_combine_fn: Optional[Callable] = None
    
    # Normalization parameters
    normalize: bool = True
    state_stats: Optional[Dict[str, Dict[str, float]]] = None
    action_stats: Optional[Dict[str, Dict[str, float]]] = None


def pose_to_6d(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose (position + quaternion) to 6D representation.
    
    Args:
        pose: Array of shape (7,) containing [x, y, z, qx, qy, qz, qw]
        
    Returns:
        Array of shape (6,) containing [x, y, z, roll, pitch, yaw]
    """
    if pose.ndim == 0:
        pose = np.atleast_1d(pose)
    
    if len(pose) >= 7:
        # Extract position (x, y, z)
        position = pose[:3]
        
        # Extract quaternion (qx, qy, qz, qw)
        qx, qy, qz, qw = pose[3:7]
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.concatenate([position, [roll, pitch, yaw]]).astype(np.float32)
    elif len(pose) >= 6:
        # Already in 6D format
        return pose[:6].astype(np.float32)
    else:
        # Fallback: pad with zeros
        result = np.zeros(6, dtype=np.float32)
        result[:len(pose)] = pose[:len(pose)]
        return result


def combine_franka_dual_arm_state(components: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Custom state combination for Franka dual-arm robot.
    
    This function defines the specific order and structure of the state vector
    for Franka dual-arm setup.
    
    State structure:
    - Left arm joints (7 DOF)
    - Right arm joints (7 DOF)
    - Left end-effector pose (6D: x, y, z, roll, pitch, yaw)
    - Right end-effector pose (6D: x, y, z, roll, pitch, yaw)
    - Left gripper position (1D)
    - Right gripper position (1D)
    
    Total: 7 + 7 + 6 + 6 + 1 + 1 = 28 dimensions
    
    Args:
        components: Dictionary mapping component paths to numpy arrays
        
    Returns:
        Combined state vector with consistent ordering
    """
    state_parts = []
    
    # Left arm joints (7 DOF for Franka)
    # 支持多种路径格式：嵌套路径（group/dataset）和直接路径
    if "left_franka_joint_states/left_franka/joint_positions" in components:
        left_joints = components["left_franka_joint_states/left_franka/joint_positions"]
        if left_joints.ndim == 0:
            left_joints = np.atleast_1d(left_joints)
        state_parts.append(left_joints[:7])  # Ensure 7 joints
    elif "left_franka_joint_states" in components:
        left_joints = components["left_franka_joint_states"]
        if left_joints.ndim == 0:
            left_joints = np.atleast_1d(left_joints)
        state_parts.append(left_joints[:7])
    elif "left_franka/joint_positions" in components:
        left_joints = components["left_franka/joint_positions"]
        if left_joints.ndim == 0:
            left_joints = np.atleast_1d(left_joints)
        state_parts.append(left_joints[:7])
    elif "left/franka/joint_positions" in components:
        left_joints = components["left/franka/joint_positions"]
        if left_joints.ndim == 0:
            left_joints = np.atleast_1d(left_joints)
        state_parts.append(left_joints[:7])
    
    # Right arm joints (7 DOF for Franka)
    if "right_franka_joint_states/right_franka/joint_positions" in components:
        right_joints = components["right_franka_joint_states/right_franka/joint_positions"]
        if right_joints.ndim == 0:
            right_joints = np.atleast_1d(right_joints)
        state_parts.append(right_joints[:7])  # Ensure 7 joints
    elif "right_franka_joint_states" in components:
        right_joints = components["right_franka_joint_states"]
        if right_joints.ndim == 0:
            right_joints = np.atleast_1d(right_joints)
        state_parts.append(right_joints[:7])
    elif "right_franka/joint_positions" in components:
        right_joints = components["right_franka/joint_positions"]
        if right_joints.ndim == 0:
            right_joints = np.atleast_1d(right_joints)
        state_parts.append(right_joints[:7])
    elif "right/franka/joint_positions" in components:
        right_joints = components["right/franka/joint_positions"]
        if right_joints.ndim == 0:
            right_joints = np.atleast_1d(right_joints)
        state_parts.append(right_joints[:7])
    
    # Left end-effector pose (6D)
    if "left_franka_eef/left_franka/eef_pose" in components:
        left_ee_pose = components["left_franka_eef/left_franka/eef_pose"]
        state_parts.append(pose_to_6d(left_ee_pose))
    elif "left_franka_eef" in components:
        left_ee_pose = components["left_franka_eef"]
        state_parts.append(pose_to_6d(left_ee_pose))
    elif "left_franka/eef_pose" in components:
        left_ee_pose = components["left_franka/eef_pose"]
        state_parts.append(pose_to_6d(left_ee_pose))
    elif "left/franka/eef_pose" in components:
        left_ee_pose = components["left/franka/eef_pose"]
        state_parts.append(pose_to_6d(left_ee_pose))
    
    # Right end-effector pose (6D)
    if "right_franka_eef/right_franka/eef_pose" in components:
        right_ee_pose = components["right_franka_eef/right_franka/eef_pose"]
        state_parts.append(pose_to_6d(right_ee_pose))
    elif "right_franka_eef" in components:
        right_ee_pose = components["right_franka_eef"]
        state_parts.append(pose_to_6d(right_ee_pose))
    elif "right_franka/eef_pose" in components:
        right_ee_pose = components["right_franka/eef_pose"]
        state_parts.append(pose_to_6d(right_ee_pose))
    elif "right/franka/eef_pose" in components:
        right_ee_pose = components["right/franka/eef_pose"]
        state_parts.append(pose_to_6d(right_ee_pose))
    
    # Left gripper position
    if "left_franka_gripper/left_franka/gripper_position" in components:
        left_gripper = components["left_franka_gripper/left_franka/gripper_position"]
        state_parts.append(np.atleast_1d(left_gripper))
    elif "left_franka_gripper" in components:
        left_gripper = components["left_franka_gripper"]
        state_parts.append(np.atleast_1d(left_gripper))
    elif "left_franka/gripper_position" in components:
        left_gripper = components["left_franka/gripper_position"]
        state_parts.append(np.atleast_1d(left_gripper))
    elif "left/franka/gripper_position" in components:
        left_gripper = components["left/franka/gripper_position"]
        state_parts.append(np.atleast_1d(left_gripper))
    
    # Right gripper position
    if "right_franka_gripper/right_franka/gripper_position" in components:
        right_gripper = components["right_franka_gripper/right_franka/gripper_position"]
        state_parts.append(np.atleast_1d(right_gripper))
    elif "right_franka_gripper" in components:
        right_gripper = components["right_franka_gripper"]
        state_parts.append(np.atleast_1d(right_gripper))
    elif "right_franka/gripper_position" in components:
        right_gripper = components["right_franka/gripper_position"]
        state_parts.append(np.atleast_1d(right_gripper))
    elif "right/franka/gripper_position" in components:
        right_gripper = components["right/franka/gripper_position"]
        state_parts.append(np.atleast_1d(right_gripper))

    # Concatenate all parts
    # Total: 7 + 7 + 6 + 6 + 1 + 1 = 28 dimensions
    if len(state_parts) == 0:
        raise ValueError("No state components found. Check HDF5 paths.")
    return np.concatenate(state_parts, axis=-1).astype(np.float32)


def combine_franka_dual_arm_action(components: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Custom action combination for Franka dual-arm robot.
    
    This function defines the specific order and structure of the action vector
    for Franka dual-arm setup.
    
    Action structure:
    - Left arm joint commands (7 DOF)
    - Right arm joint commands (7 DOF)
    - Left end-effector pose commands (6D: x, y, z, roll, pitch, yaw)
    - Right end-effector pose commands (6D: x, y, z, roll, pitch, yaw)
    - Left gripper command (1D)
    - Right gripper command (1D)
    
    Total: 7 + 7 + 6 + 6 + 1 + 1 = 28 dimensions
    
    Args:
        components: Dictionary mapping component paths to numpy arrays
        
    Returns:
        Combined action vector with consistent ordering
    """
    action_parts = []
    
    # Left arm joint commands (7 DOF)
    # 支持多种路径格式：嵌套路径（group/dataset）和直接路径
    if "left_franka_joint_states/left_franka/joint_positions" in components:
        left_commands = components["left_franka_joint_states/left_franka/joint_positions"]
        if left_commands.ndim == 0:
            left_commands = np.atleast_1d(left_commands)
        action_parts.append(left_commands[:7])
    elif "left_franka_joint_states" in components:
        left_commands = components["left_franka_joint_states"]
        if left_commands.ndim == 0:
            left_commands = np.atleast_1d(left_commands)
        action_parts.append(left_commands[:7])
    elif "left_franka/joint_positions" in components:
        left_commands = components["left_franka/joint_positions"]
        if left_commands.ndim == 0:
            left_commands = np.atleast_1d(left_commands)
        action_parts.append(left_commands[:7])
    elif "left/franka/joint_positions" in components:
        left_commands = components["left/franka/joint_positions"]
        if left_commands.ndim == 0:
            left_commands = np.atleast_1d(left_commands)
        action_parts.append(left_commands[:7])
    
    # Right arm joint commands (7 DOF)
    if "right_franka_joint_states/right_franka/joint_positions" in components:
        right_commands = components["right_franka_joint_states/right_franka/joint_positions"]
        if right_commands.ndim == 0:
            right_commands = np.atleast_1d(right_commands)
        action_parts.append(right_commands[:7])
    elif "right_franka_joint_states" in components:
        right_commands = components["right_franka_joint_states"]
        if right_commands.ndim == 0:
            right_commands = np.atleast_1d(right_commands)
        action_parts.append(right_commands[:7])
    elif "right_franka/joint_positions" in components:
        right_commands = components["right_franka/joint_positions"]
        if right_commands.ndim == 0:
            right_commands = np.atleast_1d(right_commands)
        action_parts.append(right_commands[:7])
    elif "right/franka/joint_positions" in components:
        right_commands = components["right/franka/joint_positions"]
        if right_commands.ndim == 0:
            right_commands = np.atleast_1d(right_commands)
        action_parts.append(right_commands[:7])
        
    # Left end-effector pose commands (6D)
    if "left_franka_eef/left_franka/eef_pose" in components:
        left_ee_pose = components["left_franka_eef/left_franka/eef_pose"]
        action_parts.append(pose_to_6d(left_ee_pose))
    elif "left_franka_eef" in components:
        left_ee_pose = components["left_franka_eef"]
        action_parts.append(pose_to_6d(left_ee_pose))
    elif "left_franka/eef_pose" in components:
        left_ee_pose = components["left_franka/eef_pose"]
        action_parts.append(pose_to_6d(left_ee_pose))
    elif "left/franka/eef_pose" in components:
        left_ee_pose = components["left/franka/eef_pose"]
        action_parts.append(pose_to_6d(left_ee_pose))
    
    # Right end-effector pose commands (6D)
    if "right_franka_eef/right_franka/eef_pose" in components:
        right_ee_pose = components["right_franka_eef/right_franka/eef_pose"]
        action_parts.append(pose_to_6d(right_ee_pose))
    elif "right_franka_eef" in components:
        right_ee_pose = components["right_franka_eef"]
        action_parts.append(pose_to_6d(right_ee_pose))
    elif "right_franka/eef_pose" in components:
        right_ee_pose = components["right_franka/eef_pose"]
        action_parts.append(pose_to_6d(right_ee_pose))
    elif "right/franka/eef_pose" in components:
        right_ee_pose = components["right/franka/eef_pose"]
        action_parts.append(pose_to_6d(right_ee_pose))
    
    # Left gripper command
    if "left_franka_gripper/left_franka/gripper_position" in components:
        left_gripper_cmd = components["left_franka_gripper/left_franka/gripper_position"]
        action_parts.append(np.atleast_1d(left_gripper_cmd))
    elif "left_franka_gripper" in components:
        left_gripper_cmd = components["left_franka_gripper"]
        action_parts.append(np.atleast_1d(left_gripper_cmd))
    elif "left_franka/gripper_position" in components:
        left_gripper_cmd = components["left_franka/gripper_position"]
        action_parts.append(np.atleast_1d(left_gripper_cmd))
    elif "left/franka/gripper_position" in components:
        left_gripper_cmd = components["left/franka/gripper_position"]
        action_parts.append(np.atleast_1d(left_gripper_cmd))
    
    # Right gripper command
    if "right_franka_gripper/right_franka/gripper_position" in components:
        right_gripper_cmd = components["right_franka_gripper/right_franka/gripper_position"]
        action_parts.append(np.atleast_1d(right_gripper_cmd))
    elif "right_franka_gripper" in components:
        right_gripper_cmd = components["right_franka_gripper"]
        action_parts.append(np.atleast_1d(right_gripper_cmd))
    elif "right_franka/gripper_position" in components:
        right_gripper_cmd = components["right_franka/gripper_position"]
        action_parts.append(np.atleast_1d(right_gripper_cmd))
    elif "right/franka/gripper_position" in components:
        right_gripper_cmd = components["right/franka/gripper_position"]
        action_parts.append(np.atleast_1d(right_gripper_cmd))
    
    # Concatenate all parts
    # Total: 7 + 7 + 6 + 6 + 1 + 1 = 28 dimensions
    if len(action_parts) == 0:
        raise ValueError("No action components found. Check HDF5 paths.")
    return np.concatenate(action_parts, axis=-1).astype(np.float32)




def get_state_action_mapping() -> StateActionMapping:
    """
    Main function called by the converter to get custom mapping.
    
    Modify this function to return your specific robot's mapping.
    
    Returns:
        StateActionMapping configuration for Franka dual-arm robot
    """
    
    # Define which HDF5 paths contain state data
    # HDF5 结构是嵌套的：state/left_franka_joint_states/left_franka/joint_positions
    # 但 extract_components 不支持嵌套路径，所以我们需要使用实际的 dataset 路径
    # 根据 ros2_to_lerobot_converter.py，实际的 dataset 路径是：left_franka_joint_states/left_franka/joint_positions
    state_components = [
        # 嵌套路径格式（group/dataset）- 这是实际的数据路径
        "left_franka_joint_states/left_franka/joint_positions",
        "right_franka_joint_states/right_franka/joint_positions",
        "left_franka_eef/left_franka/eef_pose",
        "right_franka_eef/right_franka/eef_pose",
        "left_franka_gripper/left_franka/gripper_position",
        "right_franka_gripper/right_franka/gripper_position",
    ]
    
    # Define which HDF5 paths contain action data
    action_components = [
        # 嵌套路径格式（group/dataset）- 这是实际的数据路径
        "left_franka_joint_states/left_franka/joint_positions",
        "right_franka_joint_states/right_franka/joint_positions",
        "left_franka_eef/left_franka/eef_pose",
        "right_franka_eef/right_franka/eef_pose",
        "left_franka_gripper/left_franka/gripper_position",
        "right_franka_gripper/right_franka/gripper_position",
    ]
    
    # Optional: Define normalization statistics
    # These would typically be computed from your training data
    state_stats = {
        "mean": np.zeros(28),  # 28-dimensional state
        "std": np.ones(28),
        "min": np.full(28, -np.inf),
        "max": np.full(28, np.inf)
    }
    
    action_stats = {
        "mean": np.zeros(28),  # 28-dimensional action
        "std": np.ones(28),
        "min": np.full(28, -np.inf),
        "max": np.full(28, np.inf)
    }
    
    return StateActionMapping(
        state_components=state_components,
        action_components=action_components,
        state_combine_fn=combine_franka_dual_arm_state,
        action_combine_fn=combine_franka_dual_arm_action,
        normalize=True,
        state_stats=state_stats,
        action_stats=action_stats
    )


if __name__ == "__main__":
    """Test the mapping functions."""
    
    # Test dual-arm mapping
    mapping = get_state_action_mapping()
    print("Dual-arm Franka Robot Mapping:")
    print(f"  State components: {len(mapping.state_components)}")
    print(f"  Action components: {len(mapping.action_components)}")
    
    # Test with dummy data
    dummy_state_components = {
        "left_franka/joint_positions": np.random.randn(7),
        "right_franka/joint_positions": np.random.randn(7),
        "left_franka/gripper_position": np.array([0.04]),
        "right_franka/gripper_position": np.array([0.04]),
        "left_franka/eef_pose": np.concatenate([
            np.random.randn(3),  # position
            np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (w, x, y, z)
        ]),
        "right_franka/eef_pose": np.concatenate([
            np.random.randn(3),  # position
            np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (w, x, y, z)
        ]),
    }
    
    combined_state = mapping.state_combine_fn(dummy_state_components)
    print(f"  Combined state shape: {combined_state.shape}")
    print(f"  State dimensions: {combined_state.shape[0]}")
    
    dummy_action_components = {
        "left_franka/joint_positions": np.random.randn(7),
        "right_franka/joint_positions": np.random.randn(7),
        "left_franka/gripper_position": np.array([0.04]),
        "right_franka/gripper_position": np.array([0.04]),
        "left_franka/eef_pose": np.concatenate([
            np.random.randn(3),  # position
            np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (w, x, y, z)
        ]),
        "right_franka/eef_pose": np.concatenate([
            np.random.randn(3),  # position
            np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (w, x, y, z)
        ]),
    }
    
    combined_action = mapping.action_combine_fn(dummy_action_components)
    print(f"  Combined action shape: {combined_action.shape}")
    print(f"  Action dimensions: {combined_action.shape[0]}")
    
    print("\nMapping test completed successfully!")

