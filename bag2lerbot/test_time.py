from pathlib import Path
import subprocess

def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def compute_total_duration(video_files):
    total = 0.0
    durations = {}
    for vf in video_files:
        vf_path = Path(vf)
        if not vf_path.exists():
            print(f"Warning: {vf_path} does not exist, skipping")
            continue
        dur = get_video_duration(vf_path)
        durations[str(vf_path)] = dur
        total += dur
    return durations, total

# ===== 交互式 shell 直接执行 =====
video_dir = Path("/workspace/lerobot_dataset/videos/observation.images.camera_right/chunk-000")
videos = sorted(video_dir.glob("*.mp4"))

durations, total = compute_total_duration(videos)

print("每个视频时长 (秒):")
for v, d in durations.items():
    print(f"  {v}: {d:.2f}s")

print(f"\n总时长: {total:.2f}s ({total/60:.2f} 分钟)")
