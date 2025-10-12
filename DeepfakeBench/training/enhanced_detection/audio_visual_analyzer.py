"""
视听一致性分析模块
实现唇音同步分析，检测"过于完美"的同步
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import os

# 可选依赖
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

try:
    from scipy import signal
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None
    pearsonr = None

logger = logging.getLogger(__name__)


class AudioVisualAnalyzer:
    """视听一致性分析器，用于检测唇音同步异常"""
    
    def __init__(self, config: Dict):
        """
        初始化视听一致性分析器
        
        Args:
            config: 配置字典，包含唇音同步分析参数
        """
        self.config = config
        self.lip_config = config.get('lip_detection', {})
        self.audio_config = config.get('audio', {})
        
        # 唇部检测参数
        self.landmark_model_path = self.lip_config.get('model_path', '')
        self.sync_threshold = self.lip_config.get('sync_threshold', 0.95)
        
        # 音频参数
        self.sample_rate = self.audio_config.get('sample_rate', 16000)
        self.mfcc_features = self.audio_config.get('mfcc_features', 13)
        
        # 初始化dlib预测器
        self.landmark_predictor = None
        self.face_detector = None
        self._initialize_dlib()
        
        logger.info(f"视听一致性分析器初始化完成 - 同步阈值: {self.sync_threshold}")
    
    def _initialize_dlib(self):
        """初始化dlib人脸检测和关键点预测器"""
        if not DLIB_AVAILABLE:
            logger.warning("dlib不可用，将跳过唇部检测")
            self.face_detector = None
            self.landmark_predictor = None
            return
            
        try:
            if self.landmark_model_path and os.path.exists(self.landmark_model_path):
                self.face_detector = dlib.get_frontal_face_detector()
                self.landmark_predictor = dlib.shape_predictor(self.landmark_model_path)
                logger.info("dlib模型加载成功")
            else:
                logger.warning("dlib模型路径不存在，将跳过唇部检测")
                self.face_detector = None
                self.landmark_predictor = None
        except Exception as e:
            logger.error(f"dlib模型加载失败: {e}")
            self.face_detector = None
            self.landmark_predictor = None
    
    def extract_lip_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        从帧中提取唇部关键点
        
        Args:
            frame: 输入帧
            
        Returns:
            Optional[np.ndarray]: 唇部关键点坐标，如果检测失败返回None
        """
        if self.face_detector is None or self.landmark_predictor is None:
            return None
        
        try:
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            faces = self.face_detector(rgb_frame, 1)
            if len(faces) == 0:
                return None
            
            # 选择最大的人脸
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # 获取关键点
            landmarks = self.landmark_predictor(rgb_frame, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # 提取唇部关键点 (68点模型中的48-67点)
            if len(landmarks) >= 68:
                lip_points = landmarks[48:68]  # 外唇轮廓
            else:
                # 如果使用81点模型，需要调整索引
                lip_points = landmarks[48:68] if len(landmarks) >= 68 else None
            
            return lip_points
            
        except Exception as e:
            logger.error(f"唇部关键点提取失败: {e}")
            return None
    
    def compute_lip_motion_features(self, lip_landmarks_sequence: List[np.ndarray]) -> Dict:
        """
        计算唇部运动特征
        
        Args:
            lip_landmarks_sequence: 唇部关键点序列
            
        Returns:
            Dict: 包含唇部运动特征的字典
        """
        if len(lip_landmarks_sequence) < 2:
            return {'motion_energy': [], 'motion_velocity': [], 'motion_acceleration': []}
        
        motion_energy = []
        motion_velocity = []
        motion_acceleration = []
        
        for i in range(len(lip_landmarks_sequence) - 1):
            if lip_landmarks_sequence[i] is None or lip_landmarks_sequence[i + 1] is None:
                continue
            
            # 计算唇部中心点
            center_prev = np.mean(lip_landmarks_sequence[i], axis=0)
            center_curr = np.mean(lip_landmarks_sequence[i + 1], axis=0)
            
            # 计算位移
            displacement = center_curr - center_prev
            velocity = np.linalg.norm(displacement)
            
            # 计算唇部形状变化
            shape_change = np.mean(np.linalg.norm(
                lip_landmarks_sequence[i + 1] - lip_landmarks_sequence[i], axis=1
            ))
            
            # 运动能量（基于形状变化）
            energy = shape_change ** 2
            
            motion_energy.append(energy)
            motion_velocity.append(velocity)
        
        # 计算加速度
        if len(motion_velocity) > 1:
            acceleration = np.diff(motion_velocity)
            motion_acceleration = acceleration.tolist()
        
        return {
            'motion_energy': motion_energy,
            'motion_velocity': motion_velocity,
            'motion_acceleration': motion_acceleration,
            'mean_energy': np.mean(motion_energy) if motion_energy else 0.0,
            'mean_velocity': np.mean(motion_velocity) if motion_velocity else 0.0
        }
    
    def extract_audio_features(self, audio_path: str) -> Optional[Dict]:
        """
        提取音频特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Optional[Dict]: 包含音频特征的字典，如果提取失败返回None
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa不可用，跳过音频特征提取")
            return None
            
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.mfcc_features)
            
            # 提取频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # 提取过零率
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # 提取RMS能量
            rms = librosa.feature.rms(y=audio)[0]
            
            # 计算音频活动检测
            frame_length = int(0.025 * sr)  # 25ms帧长
            hop_length = int(0.010 * sr)    # 10ms帧移
            
            # 使用能量阈值检测语音活动
            energy_threshold = np.mean(rms) * 0.1
            voice_activity = rms > energy_threshold
            
            return {
                'mfcc': mfcc,
                'spectral_centroids': spectral_centroids,
                'zero_crossing_rate': zcr,
                'rms_energy': rms,
                'voice_activity': voice_activity,
                'sample_rate': sr,
                'duration': len(audio) / sr
            }
            
        except Exception as e:
            logger.error(f"音频特征提取失败: {e}")
            return None
    
    def compute_lip_audio_sync(self, lip_motion_features: Dict, audio_features: Dict) -> Dict:
        """
        计算唇音同步度
        
        Args:
            lip_motion_features: 唇部运动特征
            audio_features: 音频特征
            
        Returns:
            Dict: 包含同步度分析的字典
        """
        if not lip_motion_features or not audio_features:
            return {'sync_score': 0.0, 'is_too_perfect': False, 'correlation': 0.0}
        
        try:
            # 获取唇部运动能量序列
            lip_energy = lip_motion_features.get('motion_energy', [])
            if not lip_energy:
                return {'sync_score': 0.0, 'is_too_perfect': False, 'correlation': 0.0}
            
            # 获取音频RMS能量序列
            audio_energy = audio_features.get('rms_energy', [])
            if len(audio_energy) == 0:
                return {'sync_score': 0.0, 'is_too_perfect': False, 'correlation': 0.0}
            
            # 时间对齐：将音频特征下采样到与唇部特征相同的长度
            if len(audio_energy) != len(lip_energy):
                # 使用线性插值进行时间对齐
                audio_indices = np.linspace(0, len(audio_energy) - 1, len(lip_energy))
                audio_energy_aligned = np.interp(audio_indices, 
                                               np.arange(len(audio_energy)), 
                                               audio_energy)
            else:
                audio_energy_aligned = audio_energy
            
            # 计算相关系数
            if len(lip_energy) > 1 and len(audio_energy_aligned) > 1:
                if SCIPY_AVAILABLE:
                    correlation, p_value = pearsonr(lip_energy, audio_energy_aligned)
                else:
                    # 使用numpy实现简单的相关系数计算
                    correlation = np.corrcoef(lip_energy, audio_energy_aligned)[0, 1]
                    p_value = 1.0  # 简化处理，不计算p值
            else:
                correlation = 0.0
                p_value = 1.0
            
            # 计算同步分数
            sync_score = abs(correlation)
            
            # 检测是否"过于完美"
            is_too_perfect = sync_score > self.sync_threshold
            
            # 计算同步稳定性（相关系数的方差）
            if len(lip_energy) > 10:
                # 计算滑动窗口相关系数
                window_size = min(10, len(lip_energy) // 2)
                window_correlations = []
                
                for i in range(len(lip_energy) - window_size + 1):
                    lip_window = lip_energy[i:i + window_size]
                    audio_window = audio_energy_aligned[i:i + window_size]
                    
                    if len(lip_window) > 1 and len(audio_window) > 1:
                        if SCIPY_AVAILABLE:
                            corr, _ = pearsonr(lip_window, audio_window)
                        else:
                            corr = np.corrcoef(lip_window, audio_window)[0, 1]
                        window_correlations.append(abs(corr))
                
                sync_stability = np.std(window_correlations) if window_correlations else 0.0
            else:
                sync_stability = 0.0
            
            return {
                'sync_score': sync_score,
                'is_too_perfect': is_too_perfect,
                'correlation': correlation,
                'p_value': p_value,
                'sync_stability': sync_stability,
                'lip_energy_mean': np.mean(lip_energy),
                'audio_energy_mean': np.mean(audio_energy_aligned)
            }
            
        except Exception as e:
            logger.error(f"唇音同步计算失败: {e}")
            return {'sync_score': 0.0, 'is_too_perfect': False, 'correlation': 0.0}
    
    def analyze_audio_visual_consistency(self, frames: List[np.ndarray], audio_path: str = None) -> Dict:
        """
        综合分析视听一致性
        
        Args:
            frames: 视频帧列表
            audio_path: 音频文件路径（可选）
            
        Returns:
            Dict: 包含完整分析结果的字典
        """
        logger.info(f"开始视听一致性分析，共 {len(frames)} 帧")
        
        # 提取唇部关键点序列
        lip_landmarks_sequence = []
        for frame in frames:
            landmarks = self.extract_lip_landmarks(frame)
            lip_landmarks_sequence.append(landmarks)
        
        # 计算唇部运动特征
        lip_motion_features = self.compute_lip_motion_features(lip_landmarks_sequence)
        
        # 提取音频特征（如果提供音频文件）
        audio_features = None
        if audio_path and os.path.exists(audio_path):
            audio_features = self.extract_audio_features(audio_path)
        
        # 计算唇音同步度
        sync_results = self.compute_lip_audio_sync(lip_motion_features, audio_features)
        
        # 综合评估
        results = {
            'lip_motion_features': lip_motion_features,
            'audio_features': audio_features,
            'sync_analysis': sync_results,
            'has_audio': audio_features is not None,
            'is_anomalous': sync_results.get('is_too_perfect', False)
        }
        
        logger.info(f"视听一致性分析完成 - 同步分数: {sync_results.get('sync_score', 0.0):.4f}, "
                   f"是否异常: {results['is_anomalous']}")
        
        return results
