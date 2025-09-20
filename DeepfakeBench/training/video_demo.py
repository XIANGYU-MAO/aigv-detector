"""
视频Deepfake检测脚本
支持对视频文件进行deepfake检测，通过均匀采样帧并平均检测结果
"""
import numpy as np
import cv2
import random
import yaml
import pickle
from tqdm import tqdm
from PIL import Image as pil_image
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    dlib = None
    DLIB_AVAILABLE = False
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from trainer.trainer import Trainer
from detectors import DETECTOR
from collections import defaultdict
from PIL import Image as pil_image
from imutils import face_utils
from skimage import transform as trans
import torchvision.transforms as T
import os
import sys
from os.path import join
from typing import Tuple, List
from pathlib import Path
import argparse

# 导入视频处理工具
from video_utils import extract_frames_from_video, cleanup_temp_dir, get_video_info, is_video_file

# 导入demo.py中的函数
from demo import load_detector, infer_single_image, collect_image_paths, preprocess_face, extract_aligned_face_dlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_video_deepfake(
    video_path: str,
    model,
    face_detector=None,
    landmark_predictor=None,
    num_frames: int = 10,
    temp_dir: str = None
) -> Tuple[float, List[dict]]:
    """
    检测视频是否为deepfake
    
    Args:
        video_path: 视频文件路径
        model: 训练好的检测模型
        face_detector: 人脸检测器
        landmark_predictor: 人脸关键点预测器
        num_frames: 要提取的帧数
        temp_dir: 临时目录，如果为None则自动创建
        
    Returns:
        Tuple[float, List[dict]]: (平均fake概率, 每帧检测结果列表)
    """
    print(f"开始检测视频: {video_path}")
    
    # 获取视频信息
    try:
        video_info = get_video_info(video_path)
        print(f"视频信息: {video_info}")
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return 0.0, []
    
    # 提取帧
    try:
        frames_dir, frame_indices = extract_frames_from_video(
            video_path, num_frames, temp_dir
        )
    except Exception as e:
        print(f"提取视频帧失败: {e}")
        return 0.0, []
    
    # 收集提取的图片路径
    try:
        img_paths = collect_image_paths(frames_dir)
        print(f"找到 {len(img_paths)} 个提取的帧")
    except Exception as e:
        print(f"收集图片路径失败: {e}")
        cleanup_temp_dir(frames_dir)
        return 0.0, []
    
    # 对每帧进行检测
    frame_results = []
    fake_probs = []
    
    print("开始逐帧检测...")
    for idx, img_path in enumerate(img_paths):
        try:
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[警告] 无法读取图片: {img_path}")
                continue
            
            # 进行检测
            cls, prob = infer_single_image(img, face_detector, landmark_predictor, model)
            
            frame_result = {
                'frame_path': str(img_path),
                'frame_index': frame_indices[idx] if idx < len(frame_indices) else idx,
                'prediction': int(cls),
                'fake_probability': float(prob)
            }
            frame_results.append(frame_result)
            fake_probs.append(float(prob))
            
            print(f"帧 {idx+1}/{len(img_paths)}: 预测={cls} (0=真实, 1=假), 假概率={float(prob):.4f}")
            
        except Exception as e:
            print(f"检测帧 {idx+1} 时出错: {e}")
            continue
    
    # 计算平均结果
    if fake_probs:
        avg_fake_prob = np.mean(fake_probs)
        print(f"\n检测完成!")
        print(f"提取帧数: {len(frame_results)}")
        print(f"平均假概率: {avg_fake_prob:.4f}")
        print(f"检测结果: {'可能是Deepfake' if avg_fake_prob > 0.5 else '可能是真实视频'}")
    else:
        avg_fake_prob = 0.0
        print("未能成功检测任何帧")
    
    # 清理临时文件
    if temp_dir is None:  # 如果是自动创建的临时目录，则清理
        cleanup_temp_dir(frames_dir)
    
    return avg_fake_prob, frame_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频Deepfake检测工具"
    )
    parser.add_argument(
        "--detector_config", 
        default='training/config/detector/effort.yaml',
        help="检测器配置文件路径"
    )
    parser.add_argument(
        "--weights", 
        required=True,
        help="预训练权重文件路径"
    )
    parser.add_argument(
        "--video", 
        required=True,
        help="要检测的视频文件路径"
    )
    parser.add_argument(
        "--landmark_model", 
        default=False,
        help="dlib人脸关键点模型文件路径，如果不需要人脸裁剪则为False"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=10,
        help="从视频中提取的帧数 (默认: 10)"
    )
    parser.add_argument(
        "--output_dir", 
        default=None,
        help="帧保存目录，如果不指定则使用临时目录"
    )
    parser.add_argument(
        "--keep_frames", 
        action='store_true',
        help="保留提取的帧文件"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查视频文件
    if not os.path.exists(args.video):
        print(f"错误: 视频文件不存在: {args.video}")
        return
    
    if not is_video_file(args.video):
        print(f"错误: 不支持的文件格式: {args.video}")
        return
    
    # 加载模型
    print("正在加载检测模型...")
    try:
        model = load_detector(args.detector_config, args.weights)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 加载人脸检测器
    if args.landmark_model:
        try:
            face_detector = dlib.get_frontal_face_detector()
            landmark_predictor = dlib.shape_predictor(args.landmark_model)
            print("人脸检测器加载成功")
        except Exception as e:
            print(f"加载人脸检测器失败: {e}")
            face_detector, landmark_predictor = None, None
    else:
        face_detector, landmark_predictor = None, None
        print("跳过人脸检测器加载")
    
    # 进行视频检测
    try:
        avg_prob, frame_results = detect_video_deepfake(
            video_path=args.video,
            model=model,
            face_detector=face_detector,
            landmark_predictor=landmark_predictor,
            num_frames=args.num_frames,
            temp_dir=args.output_dir
        )
        
        # 输出最终结果
        print("\n" + "="*50)
        print("最终检测结果:")
        print(f"视频文件: {args.video}")
        print(f"提取帧数: {len(frame_results)}")
        print(f"平均假概率: {avg_prob:.4f}")
        print(f"检测结论: {'可能是Deepfake视频' if avg_prob > 0.5 else '可能是真实视频'}")
        
        # 显示每帧的详细结果
        if frame_results:
            print("\n各帧检测结果:")
            for i, result in enumerate(frame_results):
                print(f"  帧 {i+1}: 假概率={result['fake_probability']:.4f}, "
                      f"预测={'假' if result['prediction'] == 1 else '真'}")
        
        # 如果指定保留帧文件
        if args.keep_frames and args.output_dir:
            print(f"\n提取的帧已保存到: {args.output_dir}")
        
    except Exception as e:
        print(f"视频检测过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
