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

"""
Usage:
    python infer.py \
        --detector_config ./training/config/detector/effort.yaml \
        --weights ../../DeepfakeBenchv2/training/weights/easy_clipl14_cdf.pth \
        --image ./id9_id6_0009.jpg \
        --landmark_model ../../DeepfakeBenchv2/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat
"""

import argparse

# 导入视频处理工具
from video_utils import extract_frames_from_video, cleanup_temp_dir, get_video_info, is_video_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, data_dict):
    data, label = data_dict['image'], data_dict['label']
    # move data to GPU
    data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
    predictions = model(data_dict, inference=True)
    return predictions


# preprocess the input image --> cropped face, resize = 256, adding a dimension of batch (output shape: 1x3x256x256)
def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=224, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark,face
    
    else:
        return None, None


def load_detector(detector_cfg: str, weights: str):
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)  # FIXME ⚠
    model.eval()
    print("[OK] Detector loaded.")
    return model


def preprocess_face(img_bgr: np.ndarray):
    """BGR → tensor"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
    ])
    return transform(pil_image.fromarray(img_rgb)).unsqueeze(0)  # 1×3×H×W


@torch.inference_mode()
def infer_single_image(
    img_bgr: np.ndarray,
    face_detector,
    landmark_predictor,
    model,
) -> Tuple[int, float]:
    """Return (cls_out, prob)"""
    if face_detector is None or landmark_predictor is None:
        face_aligned = img_bgr
    else:
        face_aligned, _, _ = extract_aligned_face_dlib(
            face_detector, landmark_predictor, img_bgr, res=224
        )

    face_tensor = preprocess_face(face_aligned).to(device)
    data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
    preds = inference(model, data)
    cls_out = preds["cls"].squeeze().cpu().numpy()   # 0/1
    prob = preds["prob"].squeeze().cpu().numpy()     # prob
    
    # 健壮的numpy数组到标量转换
    def robust_convert_to_scalar(arr):
        try:
            return arr.item()
        except:
            try:
                return arr[0]
            except:
                try:
                    return arr.tolist()
                except:
                    return float(arr)
    
    return int(robust_convert_to_scalar(cls_out)), float(robust_convert_to_scalar(prob))


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
def collect_image_paths(path_str: str) -> List[Path]:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"[Error] Path does not exist: {path_str}")

    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"[Error] Invalid image format: {p.name}")
        return [p]

    img_list = [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS]
    if not img_list:
        raise RuntimeError(f"[Error] No valid image files found in directory: {path_str}")

    return sorted(img_list)


def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake detection for images and videos"
    )
    p.add_argument("--detector_config", default='training/config/detector/effort.yaml',
                   help="YAML 配置文件路径")
    p.add_argument("--weights", required=True,
                   help="Detector 预训练权重")
    p.add_argument("--image", required=True,
                   help="tested image or video file")
    p.add_argument("--landmark_model", default=False,
                   help="dlib 81 landmarks dat 文件 / 如果不需要裁剪人脸就是False")
    p.add_argument("--num_frames", type=int, default=10,
                   help="从视频中提取的帧数 (默认: 10)")
    p.add_argument("--output_dir", default=None,
                   help="视频帧保存目录，如果不指定则使用临时目录")
    p.add_argument("--keep_frames", action='store_true',
                   help="保留提取的视频帧文件")
    return p.parse_args()


def detect_video_deepfake_simple(video_path, model, face_det, shape_predictor, num_frames=10, output_dir=None):
    """简化的视频检测函数"""
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
        frames_dir, frame_indices = extract_frames_from_video(video_path, num_frames, output_dir)
    except Exception as e:
        print(f"提取视频帧失败: {e}")
        return 0.0, []
    
    # 收集提取的图片路径
    try:
        img_paths = collect_image_paths(frames_dir)
        print(f"找到 {len(img_paths)} 个提取的帧")
    except Exception as e:
        print(f"收集图片路径失败: {e}")
        if output_dir is None:  # 如果是临时目录则清理
            cleanup_temp_dir(frames_dir)
        return 0.0, []
    
    # 对每帧进行检测
    fake_probs = []
    print("开始逐帧检测...")
    
    for idx, img_path in enumerate(img_paths):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[Warning] loading wrong，skip: {img_path}")
                continue
            
            cls, prob = infer_single_image(img, face_det, shape_predictor, model)
            fake_probs.append(float(prob))
            print(f"[{idx+1}/{len(img_paths)}] {img_path.name:>30} | Pred Label: {cls} "
                  f"(0=Real, 1=Fake) | Fake Prob: {float(prob):.4f}")
            
        except Exception as e:
            print(f"检测帧 {idx+1} 时出错: {e}")
            continue
    
    # 计算平均结果
    if fake_probs:
        avg_fake_prob = np.mean(fake_probs)
        print(f"\n检测完成!")
        print(f"提取帧数: {len(fake_probs)}")
        print(f"平均假概率: {avg_fake_prob:.4f}")
        print(f"检测结果: {'可能是Deepfake' if avg_fake_prob > 0.5 else '可能是真实视频'}")
    else:
        avg_fake_prob = 0.0
        print("未能成功检测任何帧")
    
    # 清理临时文件
    if output_dir is None:  # 如果是自动创建的临时目录，则清理
        cleanup_temp_dir(frames_dir)
    
    return avg_fake_prob, fake_probs


def main():
    args = parse_args()

    model = load_detector(args.detector_config, args.weights)
    if args.landmark_model:
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(args.landmark_model)
    else:
        face_det, shape_predictor = None, None

    # 检查输入是否为视频文件
    if is_video_file(args.image):
        print("检测到视频文件，开始视频检测...")
        avg_prob, frame_probs = detect_video_deepfake_simple(
            args.image, model, face_det, shape_predictor, 
            args.num_frames, args.output_dir
        )
        
        print("\n" + "="*50)
        print("最终检测结果:")
        print(f"视频文件: {args.image}")
        print(f"平均假概率: {avg_prob:.4f}")
        print(f"检测结论: {'可能是Deepfake视频' if avg_prob > 0.5 else '可能是真实视频'}")
        
        if args.keep_frames and args.output_dir:
            print(f"提取的帧已保存到: {args.output_dir}")
    else:
        # 原有的图片检测逻辑
        img_paths = collect_image_paths(args.image)
        multiple = len(img_paths) > 1
        if multiple:
            print(f"Collected {len(img_paths)} images in total，let's infer them...\n")

        # ---------- infer ----------
        for idx, img_path in enumerate(img_paths, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[Warning] loading wrong，skip: {img_path}", file=sys.stderr)
                continue

            cls, prob = infer_single_image(img, face_det, shape_predictor, model)
            print(
                f"[{idx}/{len(img_paths)}] {img_path.name:>30} | Pred Label: {cls} "
                f"(0=Real, 1=Fake) | Fake Prob: {float(prob):.4f}"
            )


if __name__ == "__main__":
    main()
