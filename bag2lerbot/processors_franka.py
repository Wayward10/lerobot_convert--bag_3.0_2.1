#!/usr/bin/env python3
"""
Custom Message Processors for Franka Dual-Arm Robot

Configuration is embedded in the processor file, eliminating the need for separate YAML files.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

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
        """Initialize with Franka dual-arm robot configuration."""
        self._config = self._create_config()
    
    def _create_config(self) -> ConverterConfig:
        """Create converter configuration for Franka dual-arm robot."""
        config = ConverterConfig()
        
        # Camera configuration
        config.cameras = [
            CameraConfig(
                camera_id="camera_head",
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
                camera_id="camera_left",
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
                camera_id="camera_right",
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
        config.robot_state = RobotStateConfig(
            topics=[
                TopicConfig(
                    name="/left/franka/joint_states_desired",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/franka/joint_states_desired",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/left/franka/eef_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/franka/eef_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/left/franka/gripper_state",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/franka/gripper_state",
                    type="sensor_msgs/msg/JointState",
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
        
        For standard ROS2 messages, this is usually not needed as they are
        already registered in the typestore.
        """
        # Standard ROS2 messages should already be registered
        # If you have custom messages, register them here
        pass
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Not used - this is a config provider only."""
        return {}
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Not used - this is a config provider only."""
        return [], []


class FrankaJointStateProcessor(MessageProcessor):
    """Processor for sensor_msgs/msg/JointState messages from Franka robot."""
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
        # Franka robot has 7 joints (excluding gripper)
        # Common joint name patterns for Franka
        self.joint_name_patterns = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7',
            # Alternative naming
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'
        ]
    
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
        
        # Extract joint positions
        if hasattr(msg, 'name') and hasattr(msg, 'position'):
            joint_positions = []
            joint_names_list = list(msg.name) if isinstance(msg.name, (list, tuple)) else msg.name
            
            # Try to extract joints in order
            # First, try to find joints by name pattern
            found_joints = {}
            for i, joint_name in enumerate(joint_names_list):
                for pattern_idx, pattern in enumerate(self.joint_name_patterns[:7]):
                    if pattern in joint_name.lower():
                        if pattern_idx not in found_joints:
                            found_joints[pattern_idx] = float(msg.position[i])
                            break
            
            # If we found joints by pattern, use them
            if len(found_joints) > 0:
                joint_positions = [found_joints.get(i, 0.0) for i in range(7)]
            else:
                # Fallback: use first 7 positions if available
                if len(msg.position) >= 7:
                    joint_positions = [float(msg.position[i]) for i in range(7)]
                elif len(msg.position) > 0:
                    # Pad with zeros if less than 7
                    joint_positions = [float(msg.position[i]) for i in range(len(msg.position))]
                    joint_positions.extend([0.0] * (7 - len(joint_positions)))
            
            # If we found joints, store them
            if len(joint_positions) == 7:
                data['state'][f'{arm_side}_franka/joint_positions'] = np.array(
                    joint_positions, dtype=np.float32
                )
                # For desired joint states, also store as action
                data['action'][f'{arm_side}_franka/joint_positions'] = np.array(
                    joint_positions, dtype=np.float32
                )
            elif len(joint_positions) > 0:
                # If we have some joints but not all, pad with zeros
                padded = np.zeros(7, dtype=np.float32)
                padded[:len(joint_positions)] = joint_positions
                data['state'][f'{arm_side}_franka/joint_positions'] = padded
                data['action'][f'{arm_side}_franka/joint_positions'] = padded
        
        return data
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        # Return generic fields, will be resolved based on arm_side during processing
        state_fields = ['left_franka/joint_positions', 'right_franka/joint_positions']
        action_fields = ['left_franka/joint_positions', 'right_franka/joint_positions']
        return state_fields, action_fields


class FrankaGripperStateProcessor(MessageProcessor):
    """Processor for sensor_msgs/msg/JointState messages from Franka gripper."""
    
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
        """Process a JointState message from gripper."""
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
        
        # Extract gripper position (first joint position)
        if hasattr(msg, 'position') and len(msg.position) > 0:
            gripper_position = float(msg.position[0])
            data['state'][f'{arm_side}_franka/gripper_position'] = np.array(
                [gripper_position], dtype=np.float32
            )
            # Also store as action
            data['action'][f'{arm_side}_franka/gripper_position'] = np.array(
                [gripper_position], dtype=np.float32
            )
        
        return data
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['left_franka/gripper_position', 'right_franka/gripper_position']
        action_fields = ['left_franka/gripper_position', 'right_franka/gripper_position']
        return state_fields, action_fields


class FrankaEEFPoseProcessor(MessageProcessor):
    """Processor for geometry_msgs/msg/PoseStamped messages from Franka end-effector."""
    
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
    
    def pose_to_array(self, pose: Any) -> np.ndarray:
        """
        Convert Pose message to numpy array [x, y, z, qx, qy, qz, qw].
        
        Args:
            pose: geometry_msgs/msg/Pose object
            
        Returns:
            numpy array of shape (7,)
        """
        if hasattr(pose, 'position') and hasattr(pose, 'orientation'):
            pos = pose.position
            ori = pose.orientation
            return np.array([
                float(pos.x), float(pos.y), float(pos.z),
                float(ori.x), float(ori.y), float(ori.z), float(ori.w)
            ], dtype=np.float32)
        return np.zeros(7, dtype=np.float32)
    
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
        
        # Extract pose
        if hasattr(msg, 'pose'):
            pose_array = self.pose_to_array(msg.pose)
            data['state'][f'{arm_side}_franka/eef_pose'] = pose_array
            # Also store as action
            data['action'][f'{arm_side}_franka/eef_pose'] = pose_array
        
        return data
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['left_franka/eef_pose', 'right_franka/eef_pose']
        action_fields = ['left_franka/eef_pose', 'right_franka/eef_pose']
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
        '/left/franka/joint_states_desired': FrankaJointStateProcessor('left'),
        '/right/franka/joint_states_desired': FrankaJointStateProcessor('right'),
        '/left/franka/eef_pose': FrankaEEFPoseProcessor('left'),
        '/right/franka/eef_pose': FrankaEEFPoseProcessor('right'),
        '/left/franka/gripper_state': FrankaGripperStateProcessor('left'),
        '/right/franka/gripper_state': FrankaGripperStateProcessor('right'),
        
        # Generic processors (fallback, matched by message type)
        'JointState': FrankaJointStateProcessor(),  # Will auto-detect from topic
        'PoseStamped': FrankaEEFPoseProcessor(),    # Will auto-detect from topic
    }
    
    return processors

