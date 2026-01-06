
创建的文件
1. custom_state_action_mapping_franka.py — 状态-动作映射文件
    定义如何将 HDF5 数据组合为 LeRobot 的状态和动作向量
    支持双臂 Franka（左右各 7 个关节 + 末端执行器位姿 + 夹爪）
    状态维度：28 维（7+7+6+6+1+1）

2. processors_franka.py — ROS2 消息处理器
    从 ROS2 bag 文件中提取数据
    处理 JointState（关节状态和夹爪状态）
    处理 PoseStamped（末端执行器位姿）
    自动识别左右臂（根据 topic 名称）

3. 修改了 ros2_to_lerobot_converter.py
    支持根据 topic 名称选择 processor
    向后兼容现有的 processor




使用方法
根据 README.md，转换分为两步：
第一步：ROS2 Bag 同步

python ros2_to_lerobot_converter.py batch \
  --bags-dir=/workspace/lerobot_data/franka_data \
  --output-dir=/workspace/lerobot_output \
  --custom-processor=/workspace/bag2lerbot/processors_franka.py


python ros2_to_lerobot_converter.py batch \
    --bags-dir=/workspace/lerobot_data/UR_data_20250105 \
    --output-dir=/workspace/lerobot_data/UR_data_20250105_hdf5 \
    --custom-processor=/workspace/bag2lerbot/processors_ur.py


第二步：转换为 LeRobot 数据集


python synced_to_lerobot_converter.py \
  --input-dir /workspace/lerobot_output \
  --output-dir /workspace/lerobot_dataset \
  --repo-id franka_dual_arm \
  --fps=30 \
  --robot-type=franka_dual_arm \
  --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_franka.py \
  --use-hardware-encoding \
  --vcodec h264_nvenc \
  --crf 23 \
  --batch-size 4


ur5

python synced_to_lerobot_converter.py \
    --input-dir /workspace/lerobot_data/UR_data_20250105_hdf5 \
    --output-dir /workspace/lerobot_data/UR_data_20250105_lv3.0 \
    --repo-id ur_dual_arm \
    --fps=30 \
    --robot-type=ur_dual_arm \
    --mapping-file=/workspace/bag2lerbot/custom_state_action_mapping_ur.py \
    --use-hardware-encoding \
    --vcodec h264_nvenc \
    --crf 23 \
    --batch-size 4


