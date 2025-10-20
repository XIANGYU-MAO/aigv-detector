# 增强视频Deepfake检测系统

## 概述

本系统是基于EFFORT检测器的增强版视频Deepfake检测工具，通过多模态特征融合和基于规则的决策引擎，显著提升了检测准确性和鲁棒性。

## 系统架构

```
输入视频
    ↓
智能帧采样
   - 默认使用连续帧采样 (consecutive)
   - 适合光流分析和运动检测
   - 可选择均匀分布采样 (uniform)
   ↓
自动音频提取
   - 自动从视频中提取音频轨道
   - 使用ffmpeg进行音频处理
   - 支持多种音频格式
    ↓
多模态特征提取
    ├── 视觉特征提取 (EFFORT)
    ├── 帧间分析模块 (光流/SSIM)
    ├── 视听一致性模块 (唇音同步)
    └── Hallo增强分析模块 (扩散模型/频域/注意力/时间一致性)
    ↓
特征融合与决策
    ├── 基于规则的决策引擎
    ├── 多模态特征权重融合
    └── 详细的检测报告
    ↓
基于规则的决策引擎
    ↓
最终检测结果
```
## 核心功能

### 1. 帧间分析模块
- **光流分析**: 使用Farneback算法计算帧间光流，检测异常运动模式
- **SSIM分析**: 计算结构相似性指数，识别不自然的帧间变化
- **异常检测**: 基于变化率和方差检测异常

### 2. 视听一致性模块
- **唇部检测**: 使用dlib提取唇部关键点
- **音频特征**: 提取MFCC、频谱质心等音频特征
- **同步分析**: 计算唇部运动与音频的相关性
- **异常检测**: 识别"过于完美"的同步

### 3. Hallo增强分析模块
- **扩散模型伪影检测**: 检测扩散模型特有的噪声模式、频域特征、纹理特征和边缘特征
- **增强频域分析**: 针对Hallo的频域特征进行深度分析
- **交叉注意力增强检测**: 检测Hallo的音频-视觉交叉注意力机制特征
- **时间一致性增强分析**: 检测Hallo时间对齐技术的异常模式
- **综合评估**: 多维度特征融合的Hallo检测

### 4. 基于规则的决策引擎（优化版本）
- **多规则评估**: 支持强规则和弱规则组合
- **概率推导**: 新增概率推导算法，减少"uncertain"结果
- **多层次决策**: 7层决策逻辑，从强规则到概率推导逐步判断
- **阈值优化**: 降低决策阈值，提高决策确定性
- **置信度计算**: 基于规则满足度、特征强度和概率推导计算置信度
- **决策解释**: 提供决策过程、概率推导和依据的详细说明
- **针对性防御**: 针对Hallo等特定生成器的专门规则


## 使用方法

### 1. 基本使用（推荐）

```bash
python training/enhanced_detection/enhanced_video_demo.py \
    --config training/config/enhanced_detection.yaml \
    --video /path/to/your/video.mp4 \
    --num_frames 15
```

系统会自动：
- 使用连续帧采样（适合光流分析）
- 从视频中自动提取音频进行视听一致性分析

### 2. 指定为Hallo生成的视频（使用增强检测模式）

```bash
python training/enhanced_detection/enhanced_video_demo.py \
    --config training/config/enhanced_detection.yaml \
    --video /path/to/your/video.mp4 \
    --is_hallo \
    --num_frames 15
```

系统会：
- 启用Hallo增强分析模块
- 使用扩散模型伪影检测
- 进行增强频域分析
- 执行交叉注意力检测
- 进行时间一致性分析

### 3. 自定义采样策略

```bash
# 使用连续帧采样（推荐，适合光流分析）
python training/enhanced_detection/enhanced_video_demo.py \
    --config training/config/enhanced_detection.yaml \
    --video /path/to/your/video.mp4 \
    --sampling_strategy consecutive \
    --num_frames 15

# 使用均匀分布采样（传统方式）
python training/enhanced_detection/enhanced_video_demo.py \
    --config training/config/enhanced_detection.yaml \
    --video /path/to/your/video.mp4 \
    --sampling_strategy uniform \
    --num_frames 15
```

### 3. 手动提供音频文件

```bash
python training/enhanced_detection/enhanced_video_demo.py \
    --config training/config/enhanced_detection.yaml \
    --video /path/to/your/video.mp4 \
    --audio /path/to/your/audio.wav \
    --num_frames 15
```

### 3. 保留提取帧

```bash
python training/enhanced_detection/enhanced_video_demo.py \
    --config training/config/enhanced_detection.yaml \
    --video /path/to/your/video.mp4 \
    --output_dir ./extracted_frames \
    --keep_frames
```

## 配置参数

### 基础检测器配置
```yaml
base_detector:
  config_path: "training/config/detector/effort.yaml"
  weights_path: "training/weights/effort_clip_L14_trainOn_chameleon.pth"
```

### 帧间分析配置
```yaml
inter_frame:
  optical_flow:
    method: "farneback"  # 光流算法
    flow_threshold: 0.1  # 光流强度阈值
    anomaly_threshold: 0.3  # 异常检测阈值
  ssim:
    window_size: 11  # SSIM窗口大小
    anomaly_threshold: 0.05  # SSIM异常阈值
```

### 视听一致性配置
```yaml
audio_visual:
  lip_detection:
    model_path: "preprocessing/shape_predictor_81_face_landmarks.dat"
    sync_threshold: 0.95  # 同步完美阈值
  audio:
    sample_rate: 16000  # 音频采样率
    mfcc_features: 13  # MFCC特征数量
```

### 决策引擎配置
```yaml
decision_engine:
  ai_prob_threshold: 0.6  # AI概率阈值
  flow_anomaly_threshold: 0.3  # 光流异常阈值
  sync_perfect_threshold: 0.95  # 同步完美阈值
  confidence_threshold: 0.8  # 置信度阈值
  weights:
    ai_probability: 0.4  # AI概率权重
    inter_frame_anomaly: 0.3  # 帧间异常权重
    audio_visual_anomaly: 0.3  # 视听异常权重
```

## 输出结果

### 检测报告示例
```
============================================================
增强视频Deepfake检测报告
============================================================
视频文件: /path/to/video.mp4
分析帧数: 15

基础视觉检测:
  平均假概率: 0.7234
  检测帧数: 15

帧间分析:
  综合异常分数: 0.4567
  光流异常分数: 0.5234
  SSIM异常分数: 0.3900
  是否异常: 是

视听一致性分析:
  同步分数: 0.9876
  是否过于完美: 是
  包含音频: 否
  是否异常: 是

最终决策:
  结果: ai_generated
  置信度: 0.8567
  加权分数: 0.7234

决策依据:
  - 满足AI生成视频的强规则条件

总结:
  检测结论: AI生成视频
  置信度: 0.8567
```

## 决策规则

### AI生成视频规则
1. **强规则**: AI概率 > 0.6 AND 帧间异常 AND 唇音同步过于完美
2. **弱规则**: AI概率 > 0.8 OR 帧间严重异常

### 真实视频规则
1. AI概率 < 0.3 AND 帧间正常 AND 唇音同步自然

### 不确定情况
1. 特征不足以做出明确判断

## 测试系统

运行集成测试：
```bash
python enhanced_detection/test_enhanced_detection.py
```

## 性能优化

### 1. 计算效率
- 并行处理多帧分析
- GPU加速光流和SSIM计算
- 缓存中间计算结果

### 2. 内存优化
- 流式处理避免内存溢出
- 特征压缩存储
- 及时释放中间变量

## 故障排除

### 1. 常见问题
- **dlib模型加载失败**: 检查模型文件路径和权限
- **音频文件无法读取**: 确保音频格式支持（wav, mp3, m4a等）
- **内存不足**: 减少提取帧数或启用内存优化

### 2. 调试模式
设置日志级别为DEBUG：
```yaml
logging:
  level: "DEBUG"
```

## 扩展开发

### 1. 添加新的特征提取器
继承基础类并实现相应接口：
```python
class CustomFeatureExtractor:
    def extract_features(self, data):
        # 实现特征提取逻辑
        pass
```

### 2. 自定义决策规则
修改决策引擎配置：
```yaml
decision_engine:
  custom_rules:
    - name: "custom_rule"
      conditions: ["condition1", "condition2"]
      threshold: 0.8
```

## 技术细节

### 1. 光流算法
- 支持Lucas-Kanade和Farneback算法
- 自动降级处理（LK失败时使用Farneback）
- 异常检测基于光流强度和方向变化

### 2. SSIM分析
- 使用滑动窗口计算结构相似性
- 异常检测基于SSIM变化方差
- 支持多尺度分析

### 3. 唇音同步
- 基于dlib 68点或81点关键点模型
- 音频特征包括MFCC、频谱质心、过零率
- 同步度计算使用皮尔逊相关系数