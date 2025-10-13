"""
基于规则的决策引擎
综合多模态特征，通过规则和阈值进行最终判断
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DecisionEngine:
    """基于规则的决策引擎，用于综合多模态特征进行最终判断"""
    
    def __init__(self, config: Dict):
        """
        初始化决策引擎
        
        Args:
            config: 配置字典，包含决策规则参数
        """
        self.config = config
        
        # 决策阈值参数
        self.ai_prob_threshold = config.get('ai_prob_threshold', 0.6)
        self.flow_anomaly_threshold = config.get('flow_anomaly_threshold', 0.3)
        self.sync_perfect_threshold = config.get('sync_perfect_threshold', 0.95)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
        # 权重参数
        self.weights = config.get('weights', {
            'ai_probability': 0.4,
            'inter_frame_anomaly': 0.3,
            'audio_visual_anomaly': 0.3
        })
        
        logger.info(f"决策引擎初始化完成 - AI概率阈值: {self.ai_prob_threshold}, "
                   f"光流异常阈值: {self.flow_anomaly_threshold}")
    
    def evaluate_single_rule(self, rule_name: str, conditions: Dict) -> Tuple[bool, float]:
        """
        评估单个规则
        
        Args:
            rule_name: 规则名称
            conditions: 条件字典
            
        Returns:
            Tuple[bool, float]: (是否满足条件, 置信度)
        """
        if rule_name == "high_ai_probability":
            ai_prob = conditions.get('ai_probability', 0.0)
            return ai_prob > self.ai_prob_threshold, ai_prob
        
        elif rule_name == "inter_frame_anomaly":
            flow_anomaly = conditions.get('flow_anomaly_score', 0.0)
            ssim_anomaly = conditions.get('ssim_anomaly_score', 0.0)
            combined_anomaly = (flow_anomaly + ssim_anomaly) / 2.0
            return combined_anomaly > self.flow_anomaly_threshold, combined_anomaly
        
        elif rule_name == "audio_visual_anomaly":
            sync_score = conditions.get('sync_score', 0.0)
            is_too_perfect = conditions.get('is_too_perfect', False)
            return is_too_perfect and sync_score > self.sync_perfect_threshold, sync_score
        
        elif rule_name == "very_high_ai_probability":
            ai_prob = conditions.get('ai_probability', 0.0)
            return ai_prob > 0.8, ai_prob
        
        elif rule_name == "severe_inter_frame_anomaly":
            flow_anomaly = conditions.get('flow_anomaly_score', 0.0)
            ssim_anomaly = conditions.get('ssim_anomaly_score', 0.0)
            combined_anomaly = (flow_anomaly + ssim_anomaly) / 2.0
            return combined_anomaly > 0.5, combined_anomaly
        
        elif rule_name == "low_ai_probability":
            ai_prob = conditions.get('ai_probability', 0.0)
            return ai_prob < 0.3, 1.0 - ai_prob
        
        elif rule_name == "normal_inter_frame":
            flow_anomaly = conditions.get('flow_anomaly_score', 0.0)
            ssim_anomaly = conditions.get('ssim_anomaly_score', 0.0)
            combined_anomaly = (flow_anomaly + ssim_anomaly) / 2.0
            return combined_anomaly < 0.2, 1.0 - combined_anomaly
        
        elif rule_name == "natural_audio_visual":
            sync_score = conditions.get('sync_score', 0.0)
            is_too_perfect = conditions.get('is_too_perfect', False)
            return not is_too_perfect and sync_score < 0.9, 1.0 - sync_score
        
        else:
            logger.warning(f"未知规则: {rule_name}")
            return False, 0.0
    
    def apply_decision_rules(self, features: Dict) -> Dict:
        """
        应用决策规则
        
        Args:
            features: 包含所有特征的字典
            
        Returns:
            Dict: 包含决策结果的字典
        """
        # 提取特征
        ai_probability = features.get('ai_probability', 0.0)
        inter_frame_results = features.get('inter_frame_analysis', {})
        audio_visual_results = features.get('audio_visual_analysis', {})
        
        # 构建条件字典
        conditions = {
            'ai_probability': ai_probability,
            'flow_anomaly_score': inter_frame_results.get('optical_flow', {}).get('anomaly_score', 0.0),
            'ssim_anomaly_score': inter_frame_results.get('ssim', {}).get('anomaly_score', 0.0),
            'sync_score': audio_visual_results.get('sync_analysis', {}).get('sync_score', 0.0),
            'is_too_perfect': audio_visual_results.get('sync_analysis', {}).get('is_too_perfect', False)
        }
        
        # 定义决策规则
        rules = {
            # AI生成视频的强规则
            'ai_generated_strong': [
                'high_ai_probability',
                'inter_frame_anomaly', 
                'audio_visual_anomaly'
            ],
            
            # AI生成视频的弱规则
            'ai_generated_weak': [
                'very_high_ai_probability',
                'severe_inter_frame_anomaly'
            ],
            
            # 真实视频的规则
            'real_video': [
                'low_ai_probability',
                'normal_inter_frame',
                'natural_audio_visual'
            ]
        }
        
        # 评估规则
        rule_results = {}
        for rule_group, rule_list in rules.items():
            rule_satisfied = True
            total_confidence = 0.0
            satisfied_count = 0
            
            for rule_name in rule_list:
                satisfied, confidence = self.evaluate_single_rule(rule_name, conditions)
                if satisfied:
                    satisfied_count += 1
                    total_confidence += confidence
                else:
                    rule_satisfied = False
            
            # 计算规则组置信度
            if satisfied_count > 0:
                avg_confidence = total_confidence / satisfied_count
            else:
                avg_confidence = 0.0
            
            rule_results[rule_group] = {
                'satisfied': rule_satisfied,
                'confidence': avg_confidence,
                'satisfied_count': satisfied_count,
                'total_rules': len(rule_list)
            }
        
        return rule_results
    
    def make_final_decision(self, features: Dict, is_hallo_generated: bool = False) -> Dict:
        """
        做出最终决策
        
        Args:
            features: 包含所有特征的字典
            is_hallo_generated: 是否为Hallo生成的视频
            
        Returns:
            Dict: 包含最终决策结果的字典
        """
        logger.info("开始应用决策规则")
        
        # 应用决策规则
        rule_results = self.apply_decision_rules(features)
        
        # 计算综合分数
        ai_probability = features.get('ai_probability', 0.0)
        inter_frame_anomaly = features.get('inter_frame_analysis', {}).get('combined_anomaly_score', 0.0)
        audio_visual_anomaly = 1.0 if features.get('audio_visual_analysis', {}).get('is_anomalous', False) else 0.0
        
        # 如果是Hallo生成，使用增强特征
        if is_hallo_generated:
            hallo_enhanced_score = features.get('hallo_enhanced_score', 0.0)
            diffusion_score = features.get('diffusion_score', 0.0)
            enhanced_freq_score = features.get('enhanced_freq_score', 0.0)
            enhanced_attention_score = features.get('enhanced_attention_score', 0.0)
            enhanced_temporal_score = features.get('enhanced_temporal_score', 0.0)
            
            # 针对Hallo的增强决策逻辑
            if hallo_enhanced_score > 0.6 or diffusion_score > 0.7:
                return {
                    'decision': 'ai_generated',
                    'confidence': min(1.0, hallo_enhanced_score * 1.2),
                    'reasoning': [
                        f'Hallo增强检测分数: {hallo_enhanced_score:.3f}',
                        f'扩散模型检测分数: {diffusion_score:.3f}',
                        f'增强频域分数: {enhanced_freq_score:.3f}',
                        f'增强注意力分数: {enhanced_attention_score:.3f}',
                        f'增强时间一致性分数: {enhanced_temporal_score:.3f}'
                    ]
                }
            elif hallo_enhanced_score < 0.3 and diffusion_score < 0.4:
                return {
                    'decision': 'real_video',
                    'confidence': min(1.0, (1.0 - hallo_enhanced_score) * 1.2),
                    'reasoning': [
                        f'Hallo增强检测分数较低: {hallo_enhanced_score:.3f}',
                        f'扩散模型检测分数较低: {diffusion_score:.3f}',
                        '可能为真实视频'
                    ]
                }
        
        # 加权综合分数
        weighted_score = (
            ai_probability * self.weights['ai_probability'] +
            inter_frame_anomaly * self.weights['inter_frame_anomaly'] +
            audio_visual_anomaly * self.weights['audio_visual_anomaly']
        )
        
        # 决策逻辑
        decision = "uncertain"
        confidence = 0.0
        reasoning = []
        
        # 检查AI生成视频的强规则
        if rule_results['ai_generated_strong']['satisfied']:
            decision = "ai_generated"
            confidence = rule_results['ai_generated_strong']['confidence']
            reasoning.append("满足AI生成视频的强规则条件")
        
        # 检查AI生成视频的弱规则
        elif rule_results['ai_generated_weak']['satisfied']:
            decision = "ai_generated"
            confidence = rule_results['ai_generated_weak']['confidence']
            reasoning.append("满足AI生成视频的弱规则条件")
        
        # 检查真实视频的规则
        elif rule_results['real_video']['satisfied']:
            decision = "real_video"
            confidence = rule_results['real_video']['confidence']
            reasoning.append("满足真实视频的规则条件")
        
        # 基于综合分数进行决策
        else:
            if weighted_score > self.confidence_threshold:
                decision = "ai_generated"
                confidence = weighted_score
                reasoning.append(f"综合分数 {weighted_score:.3f} 超过阈值 {self.confidence_threshold}")
            elif weighted_score < (1.0 - self.confidence_threshold):
                decision = "real_video"
                confidence = 1.0 - weighted_score
                reasoning.append(f"综合分数 {weighted_score:.3f} 低于阈值 {1.0 - self.confidence_threshold}")
            else:
                decision = "uncertain"
                confidence = 0.5
                reasoning.append("特征不足以做出明确判断")
        
        # 构建最终结果
        result = {
            'decision': decision,
            'confidence': confidence,
            'weighted_score': weighted_score,
            'reasoning': reasoning,
            'rule_results': rule_results,
            'feature_scores': {
                'ai_probability': ai_probability,
                'inter_frame_anomaly': inter_frame_anomaly,
                'audio_visual_anomaly': audio_visual_anomaly
            }
        }
        
        logger.info(f"决策完成 - 结果: {decision}, 置信度: {confidence:.3f}, 综合分数: {weighted_score:.3f}")
        
        return result
    
    def get_decision_explanation(self, result: Dict) -> str:
        """
        获取决策解释
        
        Args:
            result: 决策结果字典
            
        Returns:
            str: 决策解释文本
        """
        decision = result['decision']
        confidence = result['confidence']
        reasoning = result['reasoning']
        feature_scores = result['feature_scores']
        
        explanation = f"检测结果: {decision}\n"
        explanation += f"置信度: {confidence:.3f}\n\n"
        
        explanation += "决策依据:\n"
        for reason in reasoning:
            explanation += f"- {reason}\n"
        
        explanation += f"\n特征分数:\n"
        explanation += f"- AI概率: {feature_scores['ai_probability']:.3f}\n"
        explanation += f"- 帧间异常: {feature_scores['inter_frame_anomaly']:.3f}\n"
        explanation += f"- 视听异常: {feature_scores['audio_visual_anomaly']:.3f}\n"
        
        return explanation
