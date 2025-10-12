"""
帧间分析模块
实现光流分析和SSIM分析，用于检测帧间异常变化
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
import logging

logger = logging.getLogger(__name__)


class InterFrameAnalyzer:
    """帧间分析器，用于分析连续帧之间的变化模式"""
    
    def __init__(self, config: Dict):
        """
        初始化帧间分析器
        
        Args:
            config: 配置字典，包含光流和SSIM分析参数
        """
        self.config = config
        self.optical_flow_config = config.get('optical_flow', {})
        self.ssim_config = config.get('ssim', {})
        
        # 光流分析参数
        self.flow_method = self.optical_flow_config.get('method', 'farneback')
        self.flow_threshold = self.optical_flow_config.get('flow_threshold', 0.1)
        self.anomaly_threshold = self.optical_flow_config.get('anomaly_threshold', 0.3)
        
        # SSIM分析参数
        self.ssim_window_size = self.ssim_config.get('window_size', 11)
        self.ssim_anomaly_threshold = self.ssim_config.get('anomaly_threshold', 0.05)
        
        logger.info(f"帧间分析器初始化完成 - 光流方法: {self.flow_method}, SSIM窗口大小: {self.ssim_window_size}")
    
    def compute_optical_flow(self, frames: List[np.ndarray]) -> Dict:
        """
        计算连续帧之间的光流
        
        Args:
            frames: 视频帧列表，每个帧为numpy数组
            
        Returns:
            Dict: 包含光流分析结果的字典
        """
        if len(frames) < 2:
            logger.warning("帧数不足，无法计算光流")
            return {'flow_magnitudes': [], 'flow_directions': [], 'anomaly_score': 0.0}
        
        flow_magnitudes = []
        flow_directions = []
        
        # 转换为灰度图像
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        for i in range(len(gray_frames) - 1):
            prev_frame = gray_frames[i]
            curr_frame = gray_frames[i + 1]
            
            try:
                if self.flow_method == 'farneback':
                    # 使用Farneback算法计算密集光流
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                else:
                    # 使用Lucas-Kanade算法
                    # 首先检测角点
                    corners = cv2.goodFeaturesToTrack(prev_frame, maxCorners=100, qualityLevel=0.01, minDistance=10)
                    if corners is not None and len(corners) > 0:
                        # 计算光流
                        flow = cv2.calcOpticalFlowPyrLK(
                            prev_frame, curr_frame, corners, None,
                            winSize=(15, 15), maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                        )
                        # 计算位移
                        if flow[0] is not None and flow[1] is not None:
                            displacement = flow[0] - flow[1]
                            magnitude = np.linalg.norm(displacement, axis=1)
                            angle = np.arctan2(displacement[:, 1], displacement[:, 0])
                        else:
                            magnitude = np.array([0])
                            angle = np.array([0])
                    else:
                        # 如果没有检测到角点，使用Farneback作为备选
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            except Exception as e:
                logger.warning(f"光流计算失败: {e}，使用默认值")
                magnitude = np.array([0])
                angle = np.array([0])
            
            # 计算光流统计量
            avg_magnitude = np.mean(magnitude)
            std_magnitude = np.std(magnitude)
            max_magnitude = np.max(magnitude)
            
            flow_magnitudes.append({
                'mean': avg_magnitude,
                'std': std_magnitude,
                'max': max_magnitude,
                'raw': magnitude
            })
            
            # 计算光流方向统计
            flow_directions.append({
                'mean': np.mean(angle),
                'std': np.std(angle),
                'raw': angle
            })
        
        # 分析光流异常
        anomaly_score = self._analyze_flow_anomalies(flow_magnitudes)
        
        return {
            'flow_magnitudes': flow_magnitudes,
            'flow_directions': flow_directions,
            'anomaly_score': anomaly_score,
            'method': self.flow_method
        }
    
    def _analyze_flow_anomalies(self, flow_magnitudes: List[Dict]) -> float:
        """
        分析光流异常
        
        Args:
            flow_magnitudes: 光流强度列表
            
        Returns:
            float: 异常分数 (0-1)
        """
        if len(flow_magnitudes) < 2:
            return 0.0
        
        # 提取光流强度序列
        magnitudes = [fm['mean'] for fm in flow_magnitudes]
        
        # 计算变化率
        changes = np.abs(np.diff(magnitudes))
        change_rate = np.mean(changes) if len(changes) > 0 else 0.0
        
        # 计算变化方差
        change_variance = np.var(changes) if len(changes) > 0 else 0.0
        
        # 检测异常变化
        # 如果变化率过高或变化方差过大，认为是异常
        anomaly_score = min(1.0, (change_rate / self.flow_threshold) * 0.5 + 
                           (change_variance / (self.flow_threshold ** 2)) * 0.5)
        
        logger.debug(f"光流异常分析 - 变化率: {change_rate:.4f}, 变化方差: {change_variance:.4f}, 异常分数: {anomaly_score:.4f}")
        
        return anomaly_score
    
    def compute_frame_ssim(self, frames: List[np.ndarray]) -> Dict:
        """
        计算帧间SSIM
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict: 包含SSIM分析结果的字典
        """
        if len(frames) < 2:
            logger.warning("帧数不足，无法计算SSIM")
            return {'ssim_scores': [], 'anomaly_score': 0.0}
        
        ssim_scores = []
        
        # 转换为灰度图像
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        for i in range(len(gray_frames) - 1):
            prev_frame = gray_frames[i]
            curr_frame = gray_frames[i + 1]
            
            # 确保图像尺寸一致
            if prev_frame.shape != curr_frame.shape:
                curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
            
            # 计算SSIM
            ssim_score = ssim(prev_frame, curr_frame, 
                            win_size=self.ssim_window_size,
                            data_range=255)
            ssim_scores.append(ssim_score)
        
        # 分析SSIM异常
        anomaly_score = self._analyze_ssim_anomalies(ssim_scores)
        
        return {
            'ssim_scores': ssim_scores,
            'anomaly_score': anomaly_score,
            'mean_ssim': np.mean(ssim_scores) if ssim_scores else 0.0,
            'std_ssim': np.std(ssim_scores) if ssim_scores else 0.0
        }
    
    def _analyze_ssim_anomalies(self, ssim_scores: List[float]) -> float:
        """
        分析SSIM异常变化
        
        Args:
            ssim_scores: SSIM分数列表
            
        Returns:
            float: 异常分数 (0-1)
        """
        if len(ssim_scores) < 2:
            return 0.0
        
        # 计算SSIM变化
        ssim_changes = np.abs(np.diff(ssim_scores))
        
        # 计算变化统计
        mean_change = np.mean(ssim_changes)
        std_change = np.std(ssim_changes)
        
        # 检测异常变化
        # SSIM变化过大表示帧间差异异常
        anomaly_score = min(1.0, (mean_change / self.ssim_anomaly_threshold) * 0.7 + 
                           (std_change / self.ssim_anomaly_threshold) * 0.3)
        
        logger.debug(f"SSIM异常分析 - 平均变化: {mean_change:.4f}, 变化标准差: {std_change:.4f}, 异常分数: {anomaly_score:.4f}")
        
        return anomaly_score
    
    def analyze_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        综合分析帧间变化
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict: 包含完整分析结果的字典
        """
        logger.info(f"开始帧间分析，共 {len(frames)} 帧")
        
        # 光流分析
        flow_results = self.compute_optical_flow(frames)
        
        # SSIM分析
        ssim_results = self.compute_frame_ssim(frames)
        
        # 综合异常分数
        combined_anomaly_score = (flow_results['anomaly_score'] * 0.6 + 
                                 ssim_results['anomaly_score'] * 0.4)
        
        results = {
            'optical_flow': flow_results,
            'ssim': ssim_results,
            'combined_anomaly_score': combined_anomaly_score,
            'is_anomalous': combined_anomaly_score > self.anomaly_threshold
        }
        
        logger.info(f"帧间分析完成 - 综合异常分数: {combined_anomaly_score:.4f}, 是否异常: {results['is_anomalous']}")
        
        return results
