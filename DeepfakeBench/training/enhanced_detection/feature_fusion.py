"""
特征融合模块
整合多模态特征，为决策引擎提供统一的特征表示
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureFusion:
    """特征融合器，用于整合多模态特征"""
    
    def __init__(self, config: Dict):
        """
        初始化特征融合器
        
        Args:
            config: 配置字典，包含特征融合参数
        """
        self.config = config
        
        # 特征权重
        self.feature_weights = config.get('feature_weights', {
            'visual_features': 0.4,
            'inter_frame_features': 0.3,
            'audio_visual_features': 0.3
        })
        
        # 归一化参数
        self.normalize_features = config.get('normalize_features', True)
        
        logger.info("特征融合器初始化完成")
    
    def normalize_feature_vector(self, features: np.ndarray, feature_name: str) -> np.ndarray:
        """
        归一化特征向量
        
        Args:
            features: 特征向量
            feature_name: 特征名称
            
        Returns:
            np.ndarray: 归一化后的特征向量
        """
        if not self.normalize_features:
            return features
        
        # 避免除零
        if np.std(features) == 0:
            return features
        
        # Z-score归一化
        normalized = (features - np.mean(features)) / np.std(features)
        
        # 限制范围到[-1, 1]
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
    def extract_visual_features(self, frame_results: List[Dict]) -> Dict:
        """
        从帧检测结果中提取视觉特征
        
        Args:
            frame_results: 每帧的检测结果列表
            
        Returns:
            Dict: 包含视觉特征的字典
        """
        if not frame_results:
            return {'mean_probability': 0.0, 'std_probability': 0.0, 'max_probability': 0.0}
        
        # 提取假概率
        fake_probs = [result.get('fake_probability', 0.0) for result in frame_results]
        
        # 计算统计特征
        mean_prob = np.mean(fake_probs)
        std_prob = np.std(fake_probs)
        max_prob = np.max(fake_probs)
        min_prob = np.min(fake_probs)
        
        # 计算概率分布特征
        prob_hist, _ = np.histogram(fake_probs, bins=10, range=(0, 1))
        prob_entropy = -np.sum(prob_hist * np.log(prob_hist + 1e-8)) / np.log(len(prob_hist))
        
        # 计算趋势特征
        if len(fake_probs) > 1:
            prob_trend = np.polyfit(range(len(fake_probs)), fake_probs, 1)[0]
        else:
            prob_trend = 0.0
        
        return {
            'mean_probability': mean_prob,
            'std_probability': std_prob,
            'max_probability': max_prob,
            'min_probability': min_prob,
            'probability_entropy': prob_entropy,
            'probability_trend': prob_trend,
            'raw_probabilities': fake_probs
        }
    
    def extract_inter_frame_features(self, inter_frame_results: Dict) -> Dict:
        """
        从帧间分析结果中提取特征
        
        Args:
            inter_frame_results: 帧间分析结果
            
        Returns:
            Dict: 包含帧间特征的字典
        """
        if not inter_frame_results:
            return {'combined_anomaly_score': 0.0, 'flow_anomaly_score': 0.0, 'ssim_anomaly_score': 0.0}
        
        # 光流特征
        optical_flow = inter_frame_results.get('optical_flow', {})
        flow_anomaly = optical_flow.get('anomaly_score', 0.0)
        flow_magnitudes = optical_flow.get('flow_magnitudes', [])
        
        # SSIM特征
        ssim_results = inter_frame_results.get('ssim', {})
        ssim_anomaly = ssim_results.get('anomaly_score', 0.0)
        ssim_scores = ssim_results.get('ssim_scores', [])
        
        # 综合异常分数
        combined_anomaly = inter_frame_results.get('combined_anomaly_score', 0.0)
        
        # 提取光流统计特征
        flow_features = {}
        if flow_magnitudes:
            magnitudes = [fm.get('mean', 0.0) for fm in flow_magnitudes]
            flow_features = {
                'mean_magnitude': np.mean(magnitudes),
                'std_magnitude': np.std(magnitudes),
                'max_magnitude': np.max(magnitudes),
                'magnitude_trend': np.polyfit(range(len(magnitudes)), magnitudes, 1)[0] if len(magnitudes) > 1 else 0.0
            }
        
        # 提取SSIM统计特征
        ssim_features = {}
        if ssim_scores:
            ssim_features = {
                'mean_ssim': np.mean(ssim_scores),
                'std_ssim': np.std(ssim_scores),
                'min_ssim': np.min(ssim_scores),
                'ssim_trend': np.polyfit(range(len(ssim_scores)), ssim_scores, 1)[0] if len(ssim_scores) > 1 else 0.0
            }
        
        return {
            'combined_anomaly_score': combined_anomaly,
            'flow_anomaly_score': flow_anomaly,
            'ssim_anomaly_score': ssim_anomaly,
            'flow_features': flow_features,
            'ssim_features': ssim_features,
            'is_anomalous': inter_frame_results.get('is_anomalous', False)
        }
    
    def extract_audio_visual_features(self, audio_visual_results: Dict) -> Dict:
        """
        从视听一致性分析结果中提取特征
        
        Args:
            audio_visual_results: 视听一致性分析结果
            
        Returns:
            Dict: 包含视听特征的字典
        """
        if not audio_visual_results:
            return {'sync_score': 0.0, 'is_too_perfect': False, 'has_audio': False}
        
        # 同步分析特征
        sync_analysis = audio_visual_results.get('sync_analysis', {})
        sync_score = sync_analysis.get('sync_score', 0.0)
        is_too_perfect = sync_analysis.get('is_too_perfect', False)
        correlation = sync_analysis.get('correlation', 0.0)
        sync_stability = sync_analysis.get('sync_stability', 0.0)
        
        # 唇部运动特征
        lip_motion = audio_visual_results.get('lip_motion_features', {})
        motion_energy = lip_motion.get('motion_energy', [])
        motion_velocity = lip_motion.get('motion_velocity', [])
        
        # 提取唇部运动统计特征
        lip_features = {}
        if motion_energy:
            lip_features = {
                'mean_energy': np.mean(motion_energy),
                'std_energy': np.std(motion_energy),
                'max_energy': np.max(motion_energy),
                'energy_trend': np.polyfit(range(len(motion_energy)), motion_energy, 1)[0] if len(motion_energy) > 1 else 0.0
            }
        
        if motion_velocity:
            lip_features.update({
                'mean_velocity': np.mean(motion_velocity),
                'std_velocity': np.std(motion_velocity),
                'max_velocity': np.max(motion_velocity)
            })
        
        return {
            'sync_score': sync_score,
            'is_too_perfect': is_too_perfect,
            'correlation': correlation,
            'sync_stability': sync_stability,
            'lip_features': lip_features,
            'has_audio': audio_visual_results.get('has_audio', False),
            'is_anomalous': audio_visual_results.get('is_anomalous', False)
        }
    
    def fuse_features(self, visual_features: Dict, inter_frame_features: Dict, 
                     audio_visual_features: Dict, hallo_enhanced_features: Dict = None) -> Dict:
        """
        融合多模态特征
        
        Args:
            visual_features: 视觉特征
            inter_frame_features: 帧间特征
            audio_visual_features: 视听特征
            hallo_enhanced_features: Hallo增强特征
            
        Returns:
            Dict: 融合后的特征字典
        """
        logger.info("开始特征融合")
        
        # 提取关键特征值
        fused_features = {
            # 视觉特征
            'ai_probability': visual_features.get('mean_probability', 0.0),
            'probability_std': visual_features.get('std_probability', 0.0),
            'probability_entropy': visual_features.get('probability_entropy', 0.0),
            'probability_trend': visual_features.get('probability_trend', 0.0),
            
            # 帧间特征
            'inter_frame_anomaly': inter_frame_features.get('combined_anomaly_score', 0.0),
            'flow_anomaly': inter_frame_features.get('flow_anomaly_score', 0.0),
            'ssim_anomaly': inter_frame_features.get('ssim_anomaly_score', 0.0),
            'inter_frame_is_anomalous': inter_frame_features.get('is_anomalous', False),
            
            # 视听特征
            'sync_score': audio_visual_features.get('sync_score', 0.0),
            'is_too_perfect': audio_visual_features.get('is_too_perfect', False),
            'sync_stability': audio_visual_features.get('sync_stability', 0.0),
            'audio_visual_is_anomalous': audio_visual_features.get('is_anomalous', False),
            'has_audio': audio_visual_features.get('has_audio', False)
        }
        
        # 添加Hallo增强特征
        if hallo_enhanced_features:
            fused_features.update({
                'hallo_enhanced_score': hallo_enhanced_features.get('hallo_enhanced_score', 0.0),
                'is_hallo_enhanced': hallo_enhanced_features.get('is_hallo_enhanced', False),
                'diffusion_score': hallo_enhanced_features.get('diffusion_score', 0.0),
                'enhanced_freq_score': hallo_enhanced_features.get('enhanced_freq_score', 0.0),
                'enhanced_attention_score': hallo_enhanced_features.get('enhanced_attention_score', 0.0),
                'enhanced_temporal_score': hallo_enhanced_features.get('enhanced_temporal_score', 0.0)
            })
        else:
            fused_features.update({
                'hallo_enhanced_score': 0.0,
                'is_hallo_enhanced': False,
                'diffusion_score': 0.0,
                'enhanced_freq_score': 0.0,
                'enhanced_attention_score': 0.0,
                'enhanced_temporal_score': 0.0
            })
        
        # 计算加权综合分数
        weighted_score = (
            fused_features['ai_probability'] * self.feature_weights['visual_features'] +
            fused_features['inter_frame_anomaly'] * self.feature_weights['inter_frame_features'] +
            (1.0 if fused_features['audio_visual_is_anomalous'] else 0.0) * self.feature_weights['audio_visual_features'] +
            fused_features['hallo_enhanced_score'] * 0.3  # Hallo增强特征权重
        )
        
        fused_features['weighted_score'] = weighted_score
        
        # 添加原始特征（用于调试和分析）
        fused_features['raw_features'] = {
            'visual': visual_features,
            'inter_frame': inter_frame_features,
            'audio_visual': audio_visual_features
        }
        
        logger.info(f"特征融合完成 - 加权分数: {weighted_score:.3f}")
        
        return fused_features
    
    def create_feature_vector(self, fused_features: Dict) -> np.ndarray:
        """
        创建用于机器学习的特征向量
        
        Args:
            fused_features: 融合后的特征字典
            
        Returns:
            np.ndarray: 特征向量
        """
        # 选择数值特征
        feature_names = [
            'ai_probability',
            'probability_std', 
            'probability_entropy',
            'probability_trend',
            'inter_frame_anomaly',
            'flow_anomaly',
            'ssim_anomaly',
            'sync_score',
            'sync_stability',
            'weighted_score'
        ]
        
        # 提取特征值
        feature_values = []
        for name in feature_names:
            value = fused_features.get(name, 0.0)
            feature_values.append(value)
        
        # 添加布尔特征
        feature_values.extend([
            1.0 if fused_features.get('inter_frame_is_anomalous', False) else 0.0,
            1.0 if fused_features.get('audio_visual_is_anomalous', False) else 0.0,
            1.0 if fused_features.get('is_too_perfect', False) else 0.0,
            1.0 if fused_features.get('has_audio', False) else 0.0
        ])
        
        feature_vector = np.array(feature_values, dtype=np.float32)
        
        # 归一化特征向量
        if self.normalize_features:
            feature_vector = self.normalize_feature_vector(feature_vector, "combined")
        
        return feature_vector
