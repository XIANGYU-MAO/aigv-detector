## Base Paper:
[click to download](https://github.com/user-attachments/files/22422273/2411.15633v4.pdf)
 

## How to use
Please use python3.9

0. #### download source
ckpt link: [google drive](https://drive.google.com/drive/folders/1mJu9XGbgmTM6721xDPBGPQkm8SpHKrYp?usp=sharing), download and put it into `./DeepfakeBench/training/weights`

1. #### install:
if you use conda
```python
$ conda create -n fit5230-aigi python=3.9
$ pip install torch torchvision
```

2. #### cmake and dlib：
please use dlib .whl directly, don't compile. 
link: [google drive](https://drive.google.com/drive/folders/1f8cf_EzPzzfSsinUXIpIrRGAF-KJUPd-?usp=sharing) (if python3.9, choose 39)
```python
$ python -m pip install dlib-19.22.99-cp39-cp39-win_amd64.whl`
```

3. #### numpy:
```shell
$ pip uninstall numpy
$ pip install numpy=1.23.5
```

4. #### modify:
- models link: [google drive](https://drive.google.com/drive/folders/1vvHhHOWuQV9SwRVTA1aE0J93dBAfMDRB?usp=sharing)
- modify `./DeepfakeBench/training/detectors/effort_detector.py`
- in 49th line:
```python
# please use absoluted path
clip_model = CLIPModel.from_pretrained("models--openai--clip-vit-large-patch14")
```

5. #### run：
- images
```shell
$ sh install.sh
```
```python
$ cd DeepfakeBench
$ python training/demo.py --detector_config training/config/detector/effort.yaml --weights ./training/weights/effort_clip_L14_trainOn_chameleon.pth --image {IMAGE_PATH or IMAGE_FOLDER}
```
Note, you are processing a **face image**, please add the ``--landmark_model ./preprocessing/shape_predictor_81_face_landmarks.dat`` to **extract the facial region** for inference, as our model (trained on face deepfakes) used this face extractor for processing faces.
<img width="951" height="139" alt="image1" src="https://github.com/user-attachments/assets/2f90e2b9-b6e2-4090-a86e-ab48d12dea46" />

- video
```bash
python training/video_demo.py \
    --detector_config training/config/detector/effort.yaml \
    --weights ./training/weights/effort_clip_L14_trainOn_chameleon.pth \
    --video /path/to/your/video.mp4 \
    --landmark_model ./preprocessing/shape_predictor_81_face_landmarks.dat \
    --num_frames 15
```
<img width="949" height="511" alt="video2" src="https://github.com/user-attachments/assets/5a5bfefa-d957-43f9-abfe-437b30af91b6" />
<img width="951" height="681" alt="video1" src="https://github.com/user-attachments/assets/3fdbbdd9-b29f-4f3d-a3b2-3b818a5bd545" />



##### 参数说明:
- `--detector_config`: 检测器配置文件路径
- `--weights`: 预训练权重文件路径
- `--video` / `--image`: 要检测的视频文件路径
- `--landmark_model`: dlib人脸关键点模型文件路径（可选，用于人脸对齐）
- `--num_frames`: 从视频中提取的帧数（默认：10）
- `--output_dir`: 帧保存目录（可选，不指定则使用临时目录）
- `--keep_frames`: 保留提取的帧文件（可选）

##### 故障排除

1. **视频无法打开**: 检查视频文件格式是否支持，文件是否损坏
2. **人脸检测失败**: 尝试不使用`--landmark_model`参数，或检查模型文件路径
3. **内存不足**: 减少`--num_frames`参数的值
4. **检测结果异常**: 检查预训练权重文件是否正确加载

## Reproduction

the **detailed procedure to use DeepfakeBench to reproduce the results** of paper.

### 1. Download datasets

To reproduce the results of each deepfake dataset, download the processed datasets (have already finished preprocessing such as frame extraction and face cropping) from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). 
For evaluating more diverse fake methods (such as SimSwap, BlendFace, DeepFaceLab, etc), we can use the just-released [DF40 dataset](https://github.com/YZY-stack/DF40) (with 40 distinct forgery methods implemented).

### 2. Preprocessing (**optional**)

If we only use the processed data provided by author, we will skip this step. 

Otherwise, we need to use the following codes for doing **data preprocessing strictly following DeepfakeBench**.


### 3. Rearrangement (**optional**)

> "Rearrangment" here means that we need to **create a *JSON file* for each dataset for collecting all frames within different folders**. 
> We can refer to **DeepfakeBench** and **DF40** for the provided JSON files for each dataset.

After running the above line, we obtain the JSON files for each dataset in the `./preprocessing/dataset_json` folder. The rearranged structure organizes the data in a hierarchical manner, grouping videos based on their labels and data splits (*i.e.,* train, test, validation). Each video is represented as a dictionary entry containing relevant metadata, including file paths, labels, compression levels (if applicable), *etc*. 

### 4. Training

First, We can run the following lines to train the model:
- For multiple GPUs:
```
python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py \
--detector_path ./training/config/detector/effort.yaml \
--train_dataset FaceForensics++ \
--test_dataset Celeb-DF-v2 \
--ddp
```
- For a single GPU:
```
python3 training/train.py \
--detector_path ./training/config/detector/effort.yaml \
--train_dataset FaceForensics++ \
--test_dataset Celeb-DF-v2 \
```

### 5. Testing

Once we finish training, we can test the model on several deepfake datasets such as DF40.

```
python3 training/test.py \
--detector_path ./training/config/detector/effort.yaml \
--test_dataset simswap_ff blendface_ff uniface_ff fomm_ff deepfacelab \
--weights_path ./training/weights/{CKPT}.pth
```
Then, we obtain similar evaluation results reported in our manuscript.

