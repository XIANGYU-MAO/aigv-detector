"""
针对Hallo生成视频的增强分析模块
当用户明确知道视频是Hallo生成时，使用专门的检测策略
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import logging
from scipy import fft
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class HalloEnhancedAnalyzer:
    """针对Hallo生成视频的增强分析器"""
    
    def __init__(self, config: Dict):
        """
        初始化Hallo增强分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 针对Hallo的专门参数
        self.hallo_config = config.get('hallo_enhanced', {})
        
        # 频域分析参数
        self.freq_config = self.hallo_config.get('frequency_analysis', {})
        self.freq_sensitivity = self.freq_config.get('sensitivity', 0.8)  # 提高敏感度
        
        # 交叉注意力检测参数
        self.attention_config = self.hallo_config.get('cross_attention_detection', {})
        self.attention_sensitivity = self.attention_config.get('sensitivity', 0.7)
        
        # 时间一致性参数
        self.temporal_config = self.hallo_config.get('temporal_consistency', {})
        self.temporal_sensitivity = self.temporal_config.get('sensitivity', 0.6)
        
        # 扩散模型特征检测
        self.diffusion_config = self.hallo_config.get('diffusion_detection', {})
        self.diffusion_sensitivity = self.diffusion_config.get('sensitivity', 0.9)
        
        logger.info("Hallo增强分析器初始化完成")
    
    def analyze_hallo_enhanced_features(self, frames: List[np.ndarray]) -> Dict:
        """
        针对Hallo生成视频的增强特征分析
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict: 增强分析结果
        """
        logger.info("开始Hallo增强特征分析")
        
        # 1. 扩散模型特征检测
        diffusion_results = self._detect_diffusion_artifacts(frames)
        
        # 2. 频域增强分析
        frequency_results = self._enhanced_frequency_analysis(frames)
        
        # 3. 交叉注意力增强检测
        attention_results = self._enhanced_attention_analysis(frames)
        
        # 4. 时间一致性增强分析
        temporal_results = self._enhanced_temporal_analysis(frames)
        
        # 5. 综合评估
        overall_score = self._compute_hallo_enhanced_score(
            diffusion_results, frequency_results, attention_results, temporal_results
        )
        
        return {
            'diffusion_analysis': diffusion_results,
            'frequency_analysis': frequency_results,
            'attention_analysis': attention_results,
            'temporal_analysis': temporal_results,
            'overall_hallo_score': overall_score,
            'is_hallo_enhanced': overall_score > 0.5,
            'confidence': min(1.0, overall_score * 1.2)
        }
    
    def _detect_diffusion_artifacts(self, frames: List[np.ndarray]) -> Dict:
        """
        检测扩散模型特有的伪影
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict: 扩散模型检测结果
        """
        logger.info("检测扩散模型伪影")
        
        if len(frames) < 2:
            return {'diffusion_score': 0.0, 'artifacts_detected': False}
        
        diffusion_features = []
        
        for i, frame in enumerate(frames):
            # 转换为灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. 检测扩散模型的噪声模式
            noise_pattern = self._analyze_noise_pattern(gray)
            
            # 2. 检测扩散模型的频域特征
            freq_pattern = self._analyze_diffusion_frequency(gray)
            
            # 3. 检测扩散模型的纹理特征
            texture_pattern = self._analyze_diffusion_texture(gray)
            
            # 4. 检测扩散模型的边缘特征
            edge_pattern = self._analyze_diffusion_edges(gray)
            
            diffusion_features.append({
                'noise_score': noise_pattern,
                'frequency_score': freq_pattern,
                'texture_score': texture_pattern,
                'edge_score': edge_pattern
            })
        
        # 计算综合扩散分数
        diffusion_score = self._compute_diffusion_score(diffusion_features)
        
        return {
            'diffusion_score': diffusion_score,
            'artifacts_detected': diffusion_score > self.diffusion_sensitivity,
            'features': diffusion_features
        }
    
    def _analyze_noise_pattern(self, gray_image: np.ndarray) -> float:
        """分析噪声模式"""
        # 使用拉普拉斯算子检测高频噪声
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        noise_variance = np.var(laplacian)
        
        # 扩散模型通常有特定的噪声分布
        # 计算噪声的统计特征
        noise_skewness = self._compute_skewness(laplacian.flatten())
        noise_kurtosis = self._compute_kurtosis(laplacian.flatten())
        
        # 综合噪声特征
        noise_score = min(1.0, (noise_variance / 1000.0) * 0.4 + 
                         abs(noise_skewness) * 0.3 + 
                         abs(noise_kurtosis - 3.0) * 0.3)
        
        return noise_score
    
    def _analyze_diffusion_frequency(self, gray_image: np.ndarray) -> float:
        """分析扩散模型的频域特征"""
        # 2D FFT
        f_transform = fft.fft2(gray_image)
        f_shift = fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 检测扩散模型特有的频域模式
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # 分析不同频率区域的能量分布
        low_freq_region = magnitude_spectrum[center_h-20:center_h+20, center_w-20:center_w+20]
        mid_freq_region = magnitude_spectrum[center_h-40:center_h+40, center_w-40:center_w+40]
        high_freq_region = magnitude_spectrum
        
        # 计算频率分布特征
        low_freq_energy = np.mean(low_freq_region)
        mid_freq_energy = np.mean(mid_freq_region) - low_freq_energy
        high_freq_energy = np.mean(high_freq_region) - low_freq_energy - mid_freq_energy
        
        # 扩散模型通常有特定的频率分布
        freq_score = min(1.0, abs(high_freq_energy - mid_freq_energy) / (low_freq_energy + 1e-8))
        
        return freq_score
    
    def _analyze_diffusion_texture(self, gray_image: np.ndarray) -> float:
        """分析扩散模型的纹理特征"""
        # 使用LBP (Local Binary Pattern) 分析纹理
        lbp = self._compute_lbp(gray_image)
        
        # 计算纹理的统计特征
        texture_entropy = self._compute_entropy(lbp)
        texture_uniformity = self._compute_uniformity(lbp)
        
        # 扩散模型通常有特定的纹理模式
        texture_score = min(1.0, texture_entropy * 0.6 + (1.0 - texture_uniformity) * 0.4)
        
        return texture_score
    
    def _analyze_diffusion_edges(self, gray_image: np.ndarray) -> float:
        """分析扩散模型的边缘特征"""
        # 使用Canny边缘检测
        edges = cv2.Canny(gray_image, 50, 150)
        
        # 分析边缘的分布和特征
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 计算边缘的连续性
        edge_continuity = self._compute_edge_continuity(edges)
        
        # 扩散模型通常有特定的边缘特征
        edge_score = min(1.0, edge_density * 0.5 + edge_continuity * 0.5)
        
        return edge_score
    
    def _enhanced_frequency_analysis(self, frames: List[np.ndarray]) -> Dict:
        """
        针对Hallo的增强频域分析
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict: 增强频域分析结果
        """
        logger.info("执行Hallo增强频域分析")
        
        if len(frames) < 2:
            return {'enhanced_freq_score': 0.0, 'is_anomalous': False}
        
        enhanced_features = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. 增强的频域分析
            freq_features = self._compute_enhanced_frequency_features(gray)
            
            # 2. 频域一致性分析
            consistency_features = self._compute_frequency_consistency(gray)
            
            enhanced_features.append({
                'freq_features': freq_features,
                'consistency_features': consistency_features
            })
        
        # 计算增强频域分数
        enhanced_score = self._compute_enhanced_frequency_score(enhanced_features)
        
        return {
            'enhanced_freq_score': enhanced_score,
            'is_anomalous': enhanced_score > self.freq_sensitivity,
            'features': enhanced_features
        }
    
    def _enhanced_attention_analysis(self, frames: List[np.ndarray]) -> Dict:
        """
        针对Hallo的增强注意力分析
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict: 增强注意力分析结果
        """
        logger.info("执行Hallo增强注意力分析")
        
        if len(frames) < 3:
            return {'enhanced_attention_score': 0.0, 'is_anomalous': False}
        
        attention_features = []
        
        for i in range(len(frames) - 1):
            curr_frame = frames[i]
            next_frame = frames[i + 1]
            
            # 计算增强的注意力图
            attention_map = self._compute_enhanced_attention_map(curr_frame, next_frame)
            
            # 提取注意力特征
            attention_feature = self._extract_enhanced_attention_features(attention_map)
            attention_features.append(attention_feature)
        
        # 计算增强注意力分数
        enhanced_score = self._compute_enhanced_attention_score(attention_features)
        
        return {
            'enhanced_attention_score': enhanced_score,
            'is_anomalous': enhanced_score > self.attention_sensitivity,
            'features': attention_features
        }
    
    def _enhanced_temporal_analysis(self, frames: List[np.ndarray]) -> Dict:
        """
        针对Hallo的增强时间一致性分析
        
        Args:
            frames: 视频帧列表
            
        Returns:
            Dict: 增强时间一致性分析结果
        """
        logger.info("执行Hallo增强时间一致性分析")
        
        if len(frames) < 3:
            return {'enhanced_temporal_score': 0.0, 'is_anomalous': False}
        
        temporal_features = []
        
        for i in range(len(frames) - 2):
            frame_sequence = frames[i:i+3]
            
            # 计算增强的时间一致性特征
            temporal_feature = self._compute_enhanced_temporal_features(frame_sequence)
            temporal_features.append(temporal_feature)
        
        # 计算增强时间一致性分数
        enhanced_score = self._compute_enhanced_temporal_score(temporal_features)
        
        return {
            'enhanced_temporal_score': enhanced_score,
            'is_anomalous': enhanced_score > self.temporal_sensitivity,
            'features': temporal_features
        }
    
    def _compute_enhanced_attention_map(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """计算增强的注意力图"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 使用多种方法计算注意力图
        # 1. 光流注意力
        flow_attention = self._compute_flow_attention(gray1, gray2)
        
        # 2. 频域注意力
        freq_attention = self._compute_frequency_attention(gray1, gray2)
        
        # 3. 纹理注意力
        texture_attention = self._compute_texture_attention(gray1, gray2)
        
        # 确保所有注意力图具有相同的形状
        target_shape = gray1.shape
        if flow_attention.shape != target_shape:
            flow_attention = cv2.resize(flow_attention, (target_shape[1], target_shape[0]))
        if freq_attention.shape != target_shape:
            freq_attention = cv2.resize(freq_attention, (target_shape[1], target_shape[0]))
        if texture_attention.shape != target_shape:
            texture_attention = cv2.resize(texture_attention, (target_shape[1], target_shape[0]))
        
        # 融合多种注意力
        combined_attention = (flow_attention * 0.4 + 
                            freq_attention * 0.3 + 
                            texture_attention * 0.3)
        
        return combined_attention
    
    def _compute_flow_attention(self, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """计算光流注意力"""
        try:
            # 使用Farneback光流
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            return magnitude
        except:
            return np.zeros_like(gray1, dtype=np.float32)
    
    def _compute_frequency_attention(self, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """计算频域注意力"""
        # 计算频域差异
        fft1 = fft.fft2(gray1)
        fft2 = fft.fft2(gray2)
        
        diff_fft = np.abs(fft1 - fft2)
        attention = np.real(fft.ifft2(diff_fft))
        
        return np.abs(attention)
    
    def _compute_texture_attention(self, gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
        """计算纹理注意力"""
        # 计算纹理差异
        lbp1 = self._compute_lbp(gray1)
        lbp2 = self._compute_lbp(gray2)
        
        attention = np.abs(lbp1.astype(np.float32) - lbp2.astype(np.float32))
        
        return attention
    
    def _extract_enhanced_attention_features(self, attention_map: np.ndarray) -> Dict:
        """提取增强的注意力特征"""
        # 1. 注意力集中度
        concentration = np.max(attention_map) / (np.mean(attention_map) + 1e-8)
        
        # 2. 注意力分布熵
        attention_flat = attention_map.flatten()
        attention_flat = attention_flat / (np.sum(attention_flat) + 1e-8)
        entropy = -np.sum(attention_flat * np.log(attention_flat + 1e-8))
        
        # 3. 注意力空间相关性
        spatial_corr = self._compute_spatial_correlation(attention_map)
        
        # 4. 注意力时间稳定性
        stability = 1.0 / (np.std(attention_map) + 1e-8)
        
        return {
            'concentration': concentration,
            'entropy': entropy,
            'spatial_correlation': spatial_corr,
            'stability': stability
        }
    
    def _compute_enhanced_temporal_features(self, frame_sequence: List[np.ndarray]) -> Dict:
        """计算增强的时间一致性特征"""
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frame_sequence]
        
        # 1. 时间梯度一致性
        temporal_gradients = []
        for i in range(len(gray_frames) - 1):
            grad = cv2.Laplacian(gray_frames[i+1] - gray_frames[i], cv2.CV_64F)
            temporal_gradients.append(grad)
        
        gradient_consistency = np.corrcoef(
            temporal_gradients[0].flatten(), 
            temporal_gradients[1].flatten()
        )[0, 1] if len(temporal_gradients) >= 2 else 0.0
        
        if np.isnan(gradient_consistency):
            gradient_consistency = 0.0
        
        # 2. 时间频率一致性
        freq_consistency = self._compute_temporal_frequency_consistency(gray_frames)
        
        # 3. 时间纹理一致性
        texture_consistency = self._compute_temporal_texture_consistency(gray_frames)
        
        # 4. 时间边缘一致性
        edge_consistency = self._compute_temporal_edge_consistency(gray_frames)
        
        return {
            'gradient_consistency': gradient_consistency,
            'frequency_consistency': freq_consistency,
            'texture_consistency': texture_consistency,
            'edge_consistency': edge_consistency
        }
    
    def _compute_hallo_enhanced_score(self, diffusion_results: Dict, frequency_results: Dict, 
                                    attention_results: Dict, temporal_results: Dict) -> float:
        """计算Hallo增强综合分数"""
        # 权重分配
        weights = {
            'diffusion': 0.4,  # 扩散模型特征最重要
            'frequency': 0.25,
            'attention': 0.2,
            'temporal': 0.15
        }
        
        # 计算加权分数
        score = (
            diffusion_results.get('diffusion_score', 0.0) * weights['diffusion'] +
            frequency_results.get('enhanced_freq_score', 0.0) * weights['frequency'] +
            attention_results.get('enhanced_attention_score', 0.0) * weights['attention'] +
            temporal_results.get('enhanced_temporal_score', 0.0) * weights['temporal']
        )
        
        return min(1.0, score)
    
    # 辅助方法
    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4)
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """计算熵"""
        hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _compute_uniformity(self, data: np.ndarray) -> float:
        """计算均匀性"""
        hist, _ = np.histogram(data.flatten(), bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        return np.sum(hist ** 2)
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """计算LBP特征"""
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                binary_string = ''
                
                # 8邻域
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp[i-1, j-1] = int(binary_string, 2)
        
        return lbp
    
    def _compute_edge_continuity(self, edges: np.ndarray) -> float:
        """计算边缘连续性"""
        # 使用形态学操作检测边缘连续性
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 计算连续性比例
        continuity_ratio = np.sum(closed_edges > 0) / (np.sum(edges > 0) + 1e-8)
        
        return continuity_ratio
    
    def _compute_spatial_correlation(self, attention_map: np.ndarray) -> float:
        """计算空间相关性"""
        h_corr = np.corrcoef(attention_map[:-1, :].flatten(), attention_map[1:, :].flatten())[0, 1]
        v_corr = np.corrcoef(attention_map[:, :-1].flatten(), attention_map[:, 1:].flatten())[0, 1]
        
        if np.isnan(h_corr):
            h_corr = 0.0
        if np.isnan(v_corr):
            v_corr = 0.0
        
        return (h_corr + v_corr) / 2.0
    
    def _compute_enhanced_frequency_features(self, gray_image: np.ndarray) -> Dict:
        """计算增强的频域特征"""
        f_transform = fft.fft2(gray_image)
        f_shift = fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 计算频谱质心（使用1D投影）
        h_projection = np.sum(magnitude_spectrum, axis=1)
        v_projection = np.sum(magnitude_spectrum, axis=0)
        
        # 计算频谱质心
        spectral_centroid_h = np.sum(h_projection * np.arange(len(h_projection))) / (np.sum(h_projection) + 1e-8)
        spectral_centroid_v = np.sum(v_projection * np.arange(len(v_projection))) / (np.sum(v_projection) + 1e-8)
        
        # 提取多种频域特征
        features = {
            'spectral_centroid_h': spectral_centroid_h,
            'spectral_centroid_v': spectral_centroid_v,
            'spectral_bandwidth_h': np.sqrt(np.sum(((np.arange(len(h_projection)) - spectral_centroid_h) ** 2) * h_projection) / (np.sum(h_projection) + 1e-8)),
            'spectral_bandwidth_v': np.sqrt(np.sum(((np.arange(len(v_projection)) - spectral_centroid_v) ** 2) * v_projection) / (np.sum(v_projection) + 1e-8)),
            'spectral_rolloff': np.percentile(magnitude_spectrum, 85),
            'spectral_flux': np.sum(np.diff(magnitude_spectrum.flatten()) ** 2)
        }
        
        return features
    
    def _compute_frequency_consistency(self, gray_image: np.ndarray) -> Dict:
        """计算频域一致性特征"""
        # 分块分析频域一致性
        h, w = gray_image.shape
        block_size = 32
        
        consistency_scores = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray_image[i:i+block_size, j:j+block_size]
                fft_block = fft.fft2(block)
                magnitude = np.abs(fft_block)
                
                # 计算块的频域特征
                consistency_scores.append(np.std(magnitude))
        
        return {
            'consistency_mean': np.mean(consistency_scores),
            'consistency_std': np.std(consistency_scores)
        }
    
    def _compute_enhanced_frequency_score(self, enhanced_features: List[Dict]) -> float:
        """计算增强频域分数"""
        if not enhanced_features:
            return 0.0
        
        # 分析频域特征的变化
        freq_scores = []
        consistency_scores = []
        
        for feature in enhanced_features:
            freq_features = feature['freq_features']
            consistency_features = feature['consistency_features']
            
            # 计算频域异常分数
            freq_score = (freq_features['spectral_flux'] / 1000.0 + 
                         (freq_features['spectral_bandwidth_h'] + freq_features['spectral_bandwidth_v']) / 200.0)
            freq_scores.append(freq_score)
            
            # 计算一致性分数
            consistency_score = consistency_features['consistency_std'] / (consistency_features['consistency_mean'] + 1e-8)
            consistency_scores.append(consistency_score)
        
        # 综合分数
        enhanced_score = (np.mean(freq_scores) * 0.6 + np.mean(consistency_scores) * 0.4)
        
        return min(1.0, enhanced_score)
    
    def _compute_enhanced_attention_score(self, attention_features: List[Dict]) -> float:
        """计算增强注意力分数"""
        if not attention_features:
            return 0.0
        
        concentrations = [f['concentration'] for f in attention_features]
        entropies = [f['entropy'] for f in attention_features]
        correlations = [f['spatial_correlation'] for f in attention_features]
        stabilities = [f['stability'] for f in attention_features]
        
        # 计算注意力异常指标
        concentration_anomaly = np.std(concentrations) / (np.mean(concentrations) + 1e-8)
        entropy_anomaly = np.std(entropies) / (np.mean(entropies) + 1e-8)
        correlation_anomaly = np.std(correlations) / (np.mean(correlations) + 1e-8)
        stability_anomaly = np.std(stabilities) / (np.mean(stabilities) + 1e-8)
        
        # 综合注意力分数
        attention_score = (concentration_anomaly * 0.3 + 
                          entropy_anomaly * 0.3 + 
                          correlation_anomaly * 0.2 + 
                          stability_anomaly * 0.2)
        
        return min(1.0, attention_score)
    
    def _compute_enhanced_temporal_score(self, temporal_features: List[Dict]) -> float:
        """计算增强时间一致性分数"""
        if not temporal_features:
            return 0.0
        
        gradient_consistencies = [f['gradient_consistency'] for f in temporal_features]
        freq_consistencies = [f['frequency_consistency'] for f in temporal_features]
        texture_consistencies = [f['texture_consistency'] for f in temporal_features]
        edge_consistencies = [f['edge_consistency'] for f in temporal_features]
        
        # 计算时间一致性异常
        gradient_variance = np.var(gradient_consistencies)
        freq_variance = np.var(freq_consistencies)
        texture_variance = np.var(texture_consistencies)
        edge_variance = np.var(edge_consistencies)
        
        # 综合时间一致性分数
        temporal_score = (gradient_variance * 0.3 + 
                         freq_variance * 0.3 + 
                         texture_variance * 0.2 + 
                         edge_variance * 0.2)
        
        return min(1.0, temporal_score)
    
    def _compute_diffusion_score(self, diffusion_features: List[Dict]) -> float:
        """计算扩散模型分数"""
        if not diffusion_features:
            return 0.0
        
        noise_scores = [f['noise_score'] for f in diffusion_features]
        freq_scores = [f['frequency_score'] for f in diffusion_features]
        texture_scores = [f['texture_score'] for f in diffusion_features]
        edge_scores = [f['edge_score'] for f in diffusion_features]
        
        # 计算扩散模型特征强度
        diffusion_score = (np.mean(noise_scores) * 0.3 + 
                          np.mean(freq_scores) * 0.3 + 
                          np.mean(texture_scores) * 0.2 + 
                          np.mean(edge_scores) * 0.2)
        
        return min(1.0, diffusion_score)
    
    def _compute_temporal_frequency_consistency(self, gray_frames: List[np.ndarray]) -> float:
        """计算时间频率一致性"""
        freq_spectra = []
        
        for frame in gray_frames:
            freqs, psd = welch(frame.flatten(), nperseg=min(256, len(frame.flatten())//4))
            freq_spectra.append(psd)
        
        if len(freq_spectra) >= 2:
            consistency = np.corrcoef(freq_spectra[0], freq_spectra[1])[0, 1]
            if np.isnan(consistency):
                consistency = 0.0
        else:
            consistency = 0.0
        
        return consistency
    
    def _compute_temporal_texture_consistency(self, gray_frames: List[np.ndarray]) -> float:
        """计算时间纹理一致性"""
        lbp_features = []
        
        for frame in gray_frames:
            lbp = self._compute_lbp(frame)
            hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
            lbp_features.append(hist.astype(np.float32))
        
        if len(lbp_features) >= 2:
            consistency = np.corrcoef(lbp_features[0], lbp_features[1])[0, 1]
            if np.isnan(consistency):
                consistency = 0.0
        else:
            consistency = 0.0
        
        return consistency
    
    def _compute_temporal_edge_consistency(self, gray_frames: List[np.ndarray]) -> float:
        """计算时间边缘一致性"""
        edge_features = []
        
        for frame in gray_frames:
            edges = cv2.Canny(frame, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            edge_features.append(edge_density)
        
        if len(edge_features) >= 2:
            consistency = np.corrcoef(edge_features[0], edge_features[1])[0, 1]
            if np.isnan(consistency):
                consistency = 0.0
        else:
            consistency = 0.0
        
        return consistency
