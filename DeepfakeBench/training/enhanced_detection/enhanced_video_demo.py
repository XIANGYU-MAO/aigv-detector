"""
增强版视频Deepfake检测主程序
整合多模态特征分析和基于规则的决策引擎
"""

import os
import sys
import yaml
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import load_detector, infer_single_image, collect_image_paths, preprocess_face, extract_aligned_face_dlib
from video_utils import extract_frames_from_video, cleanup_temp_dir, get_video_info, is_video_file, extract_audio_from_video
from enhanced_detection.inter_frame_analyzer import InterFrameAnalyzer
from enhanced_detection.audio_visual_analyzer import AudioVisualAnalyzer
from enhanced_detection.decision_engine import DecisionEngine
from enhanced_detection.feature_fusion import FeatureFusion

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedVideoDetector:
    """增强版视频Deepfake检测器"""
    
    def __init__(self, config_path: str):
        """
        初始化增强检测器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._initialize_components()
        
        logger.info("增强视频检测器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            # 返回默认配置
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'base_detector': {
                'config_path': 'training/config/detector/effort.yaml',
                'weights_path': 'training/weights/effort_clip_L14_trainOn_chameleon.pth'
            },
            'inter_frame': {
                'optical_flow': {
                    'method': 'farneback',
                    'flow_threshold': 0.1,
                    'anomaly_threshold': 0.3
                },
                'ssim': {
                    'window_size': 11,
                    'anomaly_threshold': 0.05
                }
            },
            'audio_visual': {
                'lip_detection': {
                    'model_path': 'preprocessing/shape_predictor_81_face_landmarks.dat',
                    'sync_threshold': 0.95
                },
                'audio': {
                    'sample_rate': 16000,
                    'mfcc_features': 13
                }
            },
            'decision_engine': {
                'ai_prob_threshold': 0.6,
                'flow_anomaly_threshold': 0.3,
                'sync_perfect_threshold': 0.95,
                'confidence_threshold': 0.8,
                'weights': {
                    'ai_probability': 0.4,
                    'inter_frame_anomaly': 0.3,
                    'audio_visual_anomaly': 0.3
                }
            },
            'feature_fusion': {
                'feature_weights': {
                    'visual_features': 0.4,
                    'inter_frame_features': 0.3,
                    'audio_visual_features': 0.3
                },
                'normalize_features': True
            }
        }
    
    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 加载基础检测器
            base_config = self.config['base_detector']
            self.base_model = load_detector(base_config['config_path'], base_config['weights_path'])
            
            # 初始化帧间分析器
            self.inter_frame_analyzer = InterFrameAnalyzer(self.config['inter_frame'])
            
            # 初始化视听一致性分析器
            self.audio_visual_analyzer = AudioVisualAnalyzer(self.config['audio_visual'])
            
            # 初始化决策引擎
            self.decision_engine = DecisionEngine(self.config['decision_engine'])
            
            # 初始化特征融合器
            self.feature_fusion = FeatureFusion(self.config['feature_fusion'])
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def detect_video(self, video_path: str, num_frames: int = 15, 
                    audio_path: Optional[str] = None, temp_dir: Optional[str] = None,
                    sampling_strategy: str = "consecutive") -> Dict:
        """
        检测视频是否为Deepfake
        
        Args:
            video_path: 视频文件路径
            num_frames: 提取的帧数
            audio_path: 音频文件路径（可选，如果不提供则从视频中自动提取）
            temp_dir: 临时目录（可选）
            sampling_strategy: 帧采样策略 ("consecutive" 或 "uniform")
            
        Returns:
            Dict: 包含检测结果的字典
        """
        logger.info(f"开始增强检测视频: {video_path}")
        
        try:
            # 1. 提取视频帧
            frames, frame_indices = self._extract_video_frames(video_path, num_frames, temp_dir, sampling_strategy)
            
            # 2. 自动提取音频（如果未提供）
            if audio_path is None:
                audio_path = self._extract_audio_from_video(video_path)
            
            # 3. 基础视觉检测
            visual_results = self._perform_visual_detection(frames)
            
            # 4. 帧间分析
            inter_frame_results = self._perform_inter_frame_analysis(frames)
            
            # 5. 视听一致性分析
            audio_visual_results = self._perform_audio_visual_analysis(frames, audio_path)
            
            # 6. 特征融合
            fused_features = self._fuse_all_features(visual_results, inter_frame_results, audio_visual_results)
            
            # 7. 决策
            final_decision = self._make_final_decision(fused_features)
            
            # 8. 构建结果
            result = self._build_result(video_path, visual_results, inter_frame_results, 
                                      audio_visual_results, final_decision, frame_indices)
            
            logger.info(f"视频检测完成: {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"视频检测失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'video_path': video_path
            }
    
    def _extract_video_frames(self, video_path: str, num_frames: int, temp_dir: Optional[str], 
                            sampling_strategy: str = "consecutive") -> Tuple[List[np.ndarray], List[int]]:
        """提取视频帧"""
        logger.info(f"提取视频帧: {num_frames} 帧")
        
        # 使用现有的视频处理工具
        frames_dir, frame_indices = extract_frames_from_video(video_path, num_frames, temp_dir, sampling_strategy)
        
        # 读取帧
        frames = []
        for i, frame_idx in enumerate(frame_indices):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}_idx_{frame_idx:06d}.jpg")
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
        
        # 清理临时目录
        if temp_dir is None:
            cleanup_temp_dir(frames_dir)
        
        logger.info(f"成功提取 {len(frames)} 帧")
        return frames, frame_indices
    
    def _extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """从视频中提取音频"""
        logger.info("尝试从视频中提取音频")
        
        try:
            audio_path = extract_audio_from_video(video_path)
            if audio_path:
                logger.info(f"音频提取成功: {audio_path}")
                return audio_path
            else:
                logger.warning("音频提取失败，将跳过视听一致性分析")
                return None
        except Exception as e:
            logger.error(f"音频提取出错: {e}")
            return None
    
    def _perform_visual_detection(self, frames: List[np.ndarray]) -> Dict:
        """执行基础视觉检测"""
        logger.info("执行基础视觉检测")
        
        frame_results = []
        fake_probs = []
        
        for i, frame in enumerate(frames):
            try:
                # 使用基础检测器进行检测
                cls, prob = infer_single_image(frame, None, None, self.base_model)
                
                frame_result = {
                    'frame_index': i,
                    'prediction': int(cls),
                    'fake_probability': float(prob)
                }
                frame_results.append(frame_result)
                fake_probs.append(float(prob))
                
            except Exception as e:
                logger.warning(f"帧 {i} 检测失败: {e}")
                continue
        
        avg_fake_prob = np.mean(fake_probs) if fake_probs else 0.0
        
        return {
            'frame_results': frame_results,
            'fake_probabilities': fake_probs,
            'average_fake_probability': avg_fake_prob,
            'total_frames': len(frame_results)
        }
    
    def _perform_inter_frame_analysis(self, frames: List[np.ndarray]) -> Dict:
        """执行帧间分析"""
        logger.info("执行帧间分析")
        
        try:
            results = self.inter_frame_analyzer.analyze_frames(frames)
            return results
        except Exception as e:
            logger.error(f"帧间分析失败: {e}")
            return {}
    
    def _perform_audio_visual_analysis(self, frames: List[np.ndarray], audio_path: Optional[str]) -> Dict:
        """执行视听一致性分析"""
        logger.info("执行视听一致性分析")
        
        try:
            results = self.audio_visual_analyzer.analyze_audio_visual_consistency(frames, audio_path)
            return results
        except Exception as e:
            logger.error(f"视听一致性分析失败: {e}")
            return {}
    
    def _fuse_all_features(self, visual_results: Dict, inter_frame_results: Dict, 
                          audio_visual_results: Dict) -> Dict:
        """融合所有特征"""
        logger.info("融合多模态特征")
        
        # 提取视觉特征
        visual_features = self.feature_fusion.extract_visual_features(visual_results.get('frame_results', []))
        
        # 提取帧间特征
        inter_frame_features = self.feature_fusion.extract_inter_frame_features(inter_frame_results)
        
        # 提取视听特征
        audio_visual_features = self.feature_fusion.extract_audio_visual_features(audio_visual_results)
        
        # 融合特征
        fused_features = self.feature_fusion.fuse_features(
            visual_features, inter_frame_features, audio_visual_features
        )
        
        return fused_features
    
    def _make_final_decision(self, fused_features: Dict) -> Dict:
        """做出最终决策"""
        logger.info("应用决策引擎")
        
        try:
            decision_result = self.decision_engine.make_final_decision(fused_features)
            return decision_result
        except Exception as e:
            logger.error(f"决策失败: {e}")
            return {
                'decision': 'uncertain',
                'confidence': 0.0,
                'reasoning': [f'决策失败: {str(e)}']
            }
    
    def _build_result(self, video_path: str, visual_results: Dict, inter_frame_results: Dict,
                     audio_visual_results: Dict, final_decision: Dict, frame_indices: List[int]) -> Dict:
        """构建最终结果"""
        
        result = {
            'success': True,
            'video_path': video_path,
            'frame_indices': frame_indices,
            'total_frames_analyzed': len(frame_indices),
            
            # 基础检测结果
            'visual_detection': {
                'average_fake_probability': visual_results.get('average_fake_probability', 0.0),
                'frame_results': visual_results.get('frame_results', []),
                'total_frames': visual_results.get('total_frames', 0)
            },
            
            # 帧间分析结果
            'inter_frame_analysis': {
                'combined_anomaly_score': inter_frame_results.get('combined_anomaly_score', 0.0),
                'optical_flow_anomaly': inter_frame_results.get('optical_flow', {}).get('anomaly_score', 0.0),
                'ssim_anomaly': inter_frame_results.get('ssim', {}).get('anomaly_score', 0.0),
                'is_anomalous': inter_frame_results.get('is_anomalous', False)
            },
            
            # 视听一致性结果
            'audio_visual_analysis': {
                'sync_score': audio_visual_results.get('sync_analysis', {}).get('sync_score', 0.0),
                'is_too_perfect': audio_visual_results.get('sync_analysis', {}).get('is_too_perfect', False),
                'has_audio': audio_visual_results.get('has_audio', False),
                'is_anomalous': audio_visual_results.get('is_anomalous', False)
            },
            
            # 最终决策
            'final_decision': final_decision,
            
            # 总结
            'summary': {
                'decision': final_decision.get('decision', 'uncertain'),
                'confidence': final_decision.get('confidence', 0.0),
                'is_ai_generated': final_decision.get('decision') == 'ai_generated',
                'is_real_video': final_decision.get('decision') == 'real_video',
                'is_uncertain': final_decision.get('decision') == 'uncertain'
            }
        }
        
        return result
    
    def print_detection_report(self, result: Dict):
        """打印检测报告"""
        if not result.get('success', False):
            print(f"检测失败: {result.get('error', '未知错误')}")
            return
        
        print("\n" + "="*60)
        print("增强视频Deepfake检测报告")
        print("="*60)
        
        print(f"视频文件: {result['video_path']}")
        print(f"分析帧数: {result['total_frames_analyzed']}")
        
        # 基础检测结果
        visual = result['visual_detection']
        print(f"\n基础视觉检测:")
        print(f"  平均假概率: {visual['average_fake_probability']:.4f}")
        print(f"  检测帧数: {visual['total_frames']}")
        
        # 帧间分析结果
        inter_frame = result['inter_frame_analysis']
        print(f"\n帧间分析:")
        print(f"  综合异常分数: {inter_frame['combined_anomaly_score']:.4f}")
        print(f"  光流异常分数: {inter_frame['optical_flow_anomaly']:.4f}")
        print(f"  SSIM异常分数: {inter_frame['ssim_anomaly']:.4f}")
        print(f"  是否异常: {'是' if inter_frame['is_anomalous'] else '否'}")
        
        # 视听一致性结果
        audio_visual = result['audio_visual_analysis']
        print(f"\n视听一致性分析:")
        print(f"  同步分数: {audio_visual['sync_score']:.4f}")
        print(f"  是否过于完美: {'是' if audio_visual['is_too_perfect'] else '否'}")
        print(f"  包含音频: {'是' if audio_visual['has_audio'] else '否'}")
        print(f"  是否异常: {'是' if audio_visual['is_anomalous'] else '否'}")
        
        # 最终决策
        decision = result['final_decision']
        print(f"\n最终决策:")
        print(f"  结果: {decision['decision']}")
        print(f"  置信度: {decision['confidence']:.4f}")
        print(f"  加权分数: {decision.get('weighted_score', 0.0):.4f}")
        
        print(f"\n决策依据:")
        for reason in decision.get('reasoning', []):
            print(f"  - {reason}")
        
        # 总结
        summary = result['summary']
        print(f"\n总结:")
        print(f"  检测结论: {'AI生成视频' if summary['is_ai_generated'] else '真实视频' if summary['is_real_video'] else '不确定'}")
        print(f"  置信度: {summary['confidence']:.4f}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="增强版视频Deepfake检测工具")
    
    parser.add_argument("--config", required=True, help="增强检测配置文件路径")
    parser.add_argument("--video", required=True, help="要检测的视频文件路径")
    parser.add_argument("--audio", default=None, help="音频文件路径（可选）")
    parser.add_argument("--num_frames", type=int, default=15, help="提取的帧数（默认：15）")
    parser.add_argument("--sampling_strategy", choices=["consecutive", "uniform"], default="consecutive", 
                       help="帧采样策略：consecutive（连续帧，适合光流分析）或uniform（均匀分布，默认：consecutive）")
    parser.add_argument("--output_dir", default=None, help="帧保存目录（可选）")
    parser.add_argument("--keep_frames", action='store_true', help="保留提取的帧文件")
    
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
    
    # 检查音频文件（如果提供）
    if args.audio and not os.path.exists(args.audio):
        print(f"警告: 音频文件不存在: {args.audio}")
        args.audio = None
    
    try:
        # 初始化增强检测器
        detector = EnhancedVideoDetector(args.config)
        
        # 执行检测
        result = detector.detect_video(
            video_path=args.video,
            num_frames=args.num_frames,
            audio_path=args.audio,
            temp_dir=args.output_dir,
            sampling_strategy=args.sampling_strategy
        )
        
        # 打印报告
        detector.print_detection_report(result)
        
        # 如果指定保留帧文件
        if args.keep_frames and args.output_dir:
            print(f"\n提取的帧已保存到: {args.output_dir}")
        
    except Exception as e:
        print(f"检测过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
