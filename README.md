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
```shell
$ sh install.sh
```
```python
$ cd DeepfakeBench
$ python training/demo.py --detector_config training/config/detector/effort.yaml --weights ./training/weights/effort_clip_L14_trainOn_chameleon.pth --image {IMAGE_PATH or IMAGE_FOLDER}
```
Note, you are processing a **face image**, please add the ``--landmark_model ./preprocessing/shape_predictor_81_face_landmarks.dat`` to **extract the facial region** for inference, as our model (trained on face deepfakes) used this face extractor for processing faces.



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

