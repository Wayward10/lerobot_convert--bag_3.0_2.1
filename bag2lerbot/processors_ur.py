#!/usr/bin/env python3
"""
Custom Message Processors with Integrated Configuration

Configuration is now embedded in the processor file, eliminating the need for separate YAML files.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ros2_to_lerobot_converter import (
    MessageProcessor, 
    ConverterConfig,
    TopicConfig,
    CameraConfig,
    RobotStateConfig
)


class ConfigProvider(MessageProcessor):
    """
    Provides converter configuration - replaces config.yaml
    """
    
    def __init__(self):
        """Initialize with your robot configuration."""
        self._config = self._create_config()
    
    def _create_config(self) -> ConverterConfig:
        """Create converter configuration."""
        config = ConverterConfig()
        
        # Camera configuration
        config.cameras = [
            CameraConfig(
                camera_id="camera_h",
                topics=[
                    TopicConfig(
                        name="/head/color/image_raw/compressed",
                        type="sensor_msgs/msg/CompressedImage",
                        frequency=30.0,
                        compressed=True,
                        modality="rgb"
                    )
                ]
            ),
            CameraConfig(
                camera_id="camera_l",
                topics=[
                    TopicConfig(
                        name="/left/color/image_raw/compressed",
                        type="sensor_msgs/msg/CompressedImage",
                        frequency=30.0,
                        compressed=True,
                        modality="rgb"
                    )
                ]
            ),
            CameraConfig(
                camera_id="camera_r",
                topics=[
                    TopicConfig(
                        name="/right/color/image_raw/compressed",
                        type="sensor_msgs/msg/CompressedImage",
                        frequency=30.0,
                        compressed=True,
                        modality="rgb"
                    )
                ]
            )
        ]
        
        # Robot state configuration
        # Note: actual_joints contains 7 values (6 joints + 1 gripper)
        # actual_tcp_pose contains end-effector pose (6D: x, y, z, rx, ry, rz)
        config.robot_state = RobotStateConfig(
            topics=[
                TopicConfig(
                    name="/left/ur5e/actual_joints",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/ur5e/actual_joints",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/left/ur5e/actual_tcp_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/ur5e/actual_tcp_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=100.0
                )
            ]
        )
        
        # Synchronization settings
        config.sync_tolerance_ms = 50.0
        config.sync_reference = None  # Auto-select
        
        # Output settings
        config.chunk_size = 1000
        config.compression = 'gzip'
        config.compression_opts = 4
        
        return config
    
    def get_converter_config(self) -> ConverterConfig:
        """Return the converter configuration."""
        return self._config
    
    def register_custom_types(self, reader: Any, typestore: Any) -> None:
        """
        Register custom ROS2 message types.
        
        For standard ROS2 messages (sensor_msgs/msg/JointState, geometry_msgs/msg/PoseStamped),
        this is usually not needed as they are already registered in the typestore.
        """
        # Standard ROS2 messages should already be registered
        # If you have custom messages, register them here
        from pathlib import Path
        from rosbags.typesys import get_types_from_msg
        
        # Only register custom message types if needed
        msg_files = {
            # Add custom message types here if you have them
            # 'device_interfaces/msg/UrStates': '/workspace/ur/ur_ws/src/device_interfaces/msg/UrStates.msg',
        }
        
        add_types = {}
        for msg_name, msg_path in msg_files.items():
            msg_file = Path(msg_path)
            if msg_file.exists():
                try:
                    msg_text = msg_file.read_text()
                    add_types.update(get_types_from_msg(msg_text, name=msg_name))
                except Exception:
                    pass
        
        # Register custom types if any
        if add_types:
            typestore.register(add_types)
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Not used - this is a config provider only."""
        return {}
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Not used - this is a config provider only."""
        return [], []


class URJointStateProcessor(MessageProcessor):
    """Processor for sensor_msgs/msg/JointState messages from UR5e robot.
    
    Note: UR5e actual_joints contains 7 values:
    - First 6: joint angles (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
    - 7th: gripper position
    """
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
    
    def _detect_arm_side(self, topic: str = None) -> str:
        """Detect arm side from topic name or use default."""
        if self.arm_side:
            return self.arm_side
        if topic:
            if '/left/' in topic.lower():
                return 'left'
            elif '/right/' in topic.lower():
                return 'right'
        return 'left'  # Default to left
    
    def process(self, msg: Any, timestamp: int, topic: str = None) -> Dict[str, Any]:
        """Process a JointState message."""
        arm_side = self._detect_arm_side(topic)
        
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # Extract timestamp from message header if available
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
                data['timestamp'] = stamp.sec * 1_000_000_000 + stamp.nanosec
        
        # Extract joint positions (contains joints + gripper)
        # UR5e actual_joints contains 7 values: 6 joints + 1 gripper
        if hasattr(msg, 'position') and msg.position is not None and len(msg.position) > 0:
            try:
                positions = np.array(msg.position, dtype=np.float32)
                
                if len(positions) >= 7:
                    # First 6 are joints
                    joint_positions = positions[:6]
                    # 7th is gripper
                    gripper_position = positions[6:7]
                    
                    data['state'][f'{arm_side}_ur5e/joint_positions'] = joint_positions
                    data['state'][f'{arm_side}_ur5e/gripper_position'] = gripper_position
                    # Also store as action (for learning from demonstrations)
                    data['action'][f'{arm_side}_ur5e/joint_positions'] = joint_positions.copy()
                    data['action'][f'{arm_side}_ur5e/gripper_position'] = gripper_position.copy()
                elif len(positions) >= 6:
                    # Only joints, no gripper (shouldn't happen for UR5e, but handle gracefully)
                    joint_positions = positions[:6]
                    data['state'][f'{arm_side}_ur5e/joint_positions'] = joint_positions
                    data['action'][f'{arm_side}_ur5e/joint_positions'] = joint_positions.copy()
                    # Set gripper to zero if missing
                    data['state'][f'{arm_side}_ur5e/gripper_position'] = np.array([0.0], dtype=np.float32)
                    data['action'][f'{arm_side}_ur5e/gripper_position'] = np.array([0.0], dtype=np.float32)
                elif len(positions) > 0:
                    # Less than 6 joints - pad with zeros
                    joint_positions = np.zeros(6, dtype=np.float32)
                    joint_positions[:len(positions)] = positions[:len(positions)]
                    data['state'][f'{arm_side}_ur5e/joint_positions'] = joint_positions
                    data['action'][f'{arm_side}_ur5e/joint_positions'] = joint_positions.copy()
                    data['state'][f'{arm_side}_ur5e/gripper_position'] = np.array([0.0], dtype=np.float32)
                    data['action'][f'{arm_side}_ur5e/gripper_position'] = np.array([0.0], dtype=np.float32)
            except Exception as e:
                # If extraction fails, log the error but continue
                # This prevents empty data points from being added to streams
                logging.warning(f"Failed to extract joint positions from {topic}: {e}")
                pass
        
        # Only return data if we have valid state/action data
        # This prevents empty data points from being added to streams
        if not data['state'] and not data['action']:
            return None
        
        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        # Return generic fields, will be resolved based on arm_side during processing
        state_fields = [
            'left_ur5e/joint_positions', 'left_ur5e/gripper_position',
            'right_ur5e/joint_positions', 'right_ur5e/gripper_position'
        ]
        action_fields = [
            'left_ur5e/joint_positions', 'left_ur5e/gripper_position',
            'right_ur5e/joint_positions', 'right_ur5e/gripper_position'
        ]
        return state_fields, action_fields


class UREEFPoseProcessor(MessageProcessor):
    """Processor for geometry_msgs/msg/PoseStamped messages from UR5e end-effector.
    
    Extracts end-effector pose (x, y, z, rx, ry, rz) from quaternion.
    """
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
    
    def _detect_arm_side(self, topic: str = None) -> str:
        """Detect arm side from topic name or use default."""
        if self.arm_side:
            return self.arm_side
        if topic:
            if '/left/' in topic.lower():
                return 'left'
            elif '/right/' in topic.lower():
                return 'right'
        return 'left'  # Default to left
    
    def process(self, msg: Any, timestamp: int, topic: str = None) -> Dict[str, Any]:
        """Process a PoseStamped message."""
        arm_side = self._detect_arm_side(topic)
        
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # Extract timestamp from message header if available
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
                data['timestamp'] = stamp.sec * 1_000_000_000 + stamp.nanosec
        
        # Extract position (x, y, z)
        try:
            if hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
                pos = msg.pose.position
                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                
                # Extract orientation and convert quaternion to euler angles
                if hasattr(msg.pose, 'orientation'):
                    orient = msg.pose.orientation
                    # Convert quaternion (qx, qy, qz, qw) to euler (rx, ry, rz)
                    euler = self._quaternion_to_euler(orient.x, orient.y, orient.z, orient.w)
                    
                    # Combine position and orientation (6D: x, y, z, rx, ry, rz)
                    eef_pose = np.concatenate([position, euler], dtype=np.float32)
                    data['state'][f'{arm_side}_ur5e/eef_pose'] = eef_pose
                    # Also store as action (for learning from demonstrations)
                    data['action'][f'{arm_side}_ur5e/eef_pose'] = eef_pose.copy()
        except Exception as e:
            # If extraction fails, return empty data (will be filtered out)
            pass
        
        # Only return data if we have valid state/action data
        # This prevents empty data points from being added to streams
        if not data['state'] and not data['action']:
            return None
        
        return data
    
    def _quaternion_to_euler(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw) - ZYX order."""
        import math
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['left_ur5e/eef_pose', 'right_ur5e/eef_pose']
        action_fields = ['left_ur5e/eef_pose', 'right_ur5e/eef_pose']
        return state_fields, action_fields


class UrStatesProcessor(MessageProcessor):
    """Processor for device_interfaces/msg/UrStates messages (legacy support)."""
    
    def process(self, msg: Any, timestamp: int, topic: str = None) -> Dict[str, Any]:
        """Process a UrStates message."""
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # Extract end-effector pose
        if hasattr(msg, 'eef_pos') and hasattr(msg.eef_pos, 'positions'):
            eef_positions = msg.eef_pos.positions
            if len(eef_positions) == 6:
                data['state']['end_eff'] = np.array(eef_positions[:6], dtype=np.float32)

        # Extract joint angles
        if hasattr(msg, 'joints') and hasattr(msg.joints, 'angles'):
            data['state']['joint_positions'] = np.array(msg.joints.angles, dtype=np.float32)
            
        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['end_eff', 'joint_position']
        action_fields = []
        return state_fields, action_fields


class GripperStateProcessor(MessageProcessor):
    """Processor for dh_gripper_driver/msg/GripperState messages."""
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Process a GripperState message."""
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # State information
        if hasattr(msg, 'position'):
            data['state']['gripper_position'] = float(msg.position)

        # Action information
        if hasattr(msg, 'target_position'):
            data['action']['gripper_position'] = float(msg.target_position)

        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['gripper_position']
        action_fields = ['gripper_position']
        return state_fields, action_fields


def get_message_processors() -> Dict[str, MessageProcessor]:
    """
    Factory function to create and return message processors.
    
    Returns:
        Dictionary mapping message type names or topic patterns to processor instances
    """
    processors = {
        # Config provider (REQUIRED - provides configuration)
        'ConfigProvider': ConfigProvider(),
        
        # Topic-specific processors (will be matched by topic name in converter)
        # These take priority over message type matching
        '/left/ur5e/actual_joints': URJointStateProcessor('left'),
        '/right/ur5e/actual_joints': URJointStateProcessor('right'),
        '/left/ur5e/actual_tcp_pose': UREEFPoseProcessor('left'),
        '/right/ur5e/actual_tcp_pose': UREEFPoseProcessor('right'),
        
        # Generic processors (fallback, matched by message type)
        # These are used if topic-specific processors don't match
        'JointState': URJointStateProcessor(),  # Will auto-detect from topic
        'PoseStamped': UREEFPoseProcessor(),    # Will auto-detect from topic
        
        # Legacy/custom processors (for backward compatibility)
        'UrStates': UrStatesProcessor(),
        'GripperState': GripperStateProcessor(),
    }
    
    return processors

