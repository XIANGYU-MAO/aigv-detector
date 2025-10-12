"""
视频处理工具模块
用于从视频中提取帧并进行deepfake检测
"""
import cv2
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def extract_frames_from_video(video_path: str, num_frames: int = 10, output_dir: str = None, 
                            sampling_strategy: str = "uniform") -> Tuple[str, List[int]]:
    """
    从视频中提取指定数量的帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数
        output_dir: 输出目录，如果为None则创建临时目录
        sampling_strategy: 采样策略 ("uniform" 或 "consecutive")
        
    Returns:
        Tuple[str, List[int]]: (输出目录路径, 提取的帧索引列表)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频信息: 总帧数={total_frames}, FPS={fps:.2f}, 时长={duration:.2f}秒")
    
    if total_frames == 0:
        cap.release()
        raise ValueError("视频文件没有有效帧")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # 根据策略计算要提取的帧索引
    if sampling_strategy == "consecutive":
        # 连续帧采样：从视频中间开始提取连续帧
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            start_idx = max(0, (total_frames - num_frames) // 2)
            frame_indices = list(range(start_idx, start_idx + num_frames))
    else:
        # 均匀分布采样（默认）
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    print(f"采样策略: {sampling_strategy}, 提取帧索引: {frame_indices}")
    
    # 提取帧并保存
    extracted_frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # 保存帧
            frame_filename = f"frame_{i:04d}_idx_{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_idx)
            print(f"已保存帧 {i+1}/{len(frame_indices)}: {frame_filename}")
        else:
            print(f"警告: 无法读取第 {frame_idx} 帧")
    
    cap.release()
    
    if not extracted_frames:
        raise RuntimeError("未能提取任何有效帧")
    
    print(f"成功提取 {len(extracted_frames)} 帧到目录: {output_dir}")
    return output_dir, extracted_frames


def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> Optional[str]:
    """
    从视频中提取音频
    
    Args:
        video_path: 视频文件路径
        output_audio_path: 输出音频文件路径，如果为None则创建临时文件
        
    Returns:
        Optional[str]: 音频文件路径，如果提取失败返回None
    """
    try:
        import subprocess
        
        if output_audio_path is None:
            # 创建临时音频文件
            temp_dir = tempfile.mkdtemp(prefix="video_audio_")
            output_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        # 使用ffmpeg提取音频
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn',  # 不处理视频
            '-acodec', 'pcm_s16le',  # 音频编码
            '-ar', '16000',  # 采样率
            '-ac', '1',  # 单声道
            '-y',  # 覆盖输出文件
            output_audio_path
        ]
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_audio_path):
            print(f"音频提取成功: {output_audio_path}")
            return output_audio_path
        else:
            print(f"音频提取失败: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("警告: ffmpeg未找到，无法提取音频")
        return None
    except Exception as e:
        print(f"音频提取出错: {e}")
        return None


def cleanup_temp_dir(temp_dir: str):
    """
    清理临时目录
    
    Args:
        temp_dir: 要清理的临时目录路径
    """
    try:
        if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")
    except Exception as e:
        print(f"清理临时目录时出错: {e}")


def get_video_info(video_path: str) -> dict:
    """
    获取视频文件信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        dict: 包含视频信息的字典
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['total_frames'] / info['fps']
    
    cap.release()
    return info


def is_video_file(file_path: str) -> bool:
    """
    检查文件是否为支持的视频格式
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否为视频文件
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    return Path(file_path).suffix.lower() in video_extensions
