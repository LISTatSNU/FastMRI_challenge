# 2023 baby unet
2023 SNU FastMRI challenge

## 0. quick start
//TODO 서버 접속하고 난 뒤 quick start 작성

## 1. 폴더 계층

### 폴더의 전체 구조
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/b551e277-4134-41bb-9d1a-8275a65c1eb7)
* FastMRI_challenge, Data, result 폴더가 위의 구조대로 설정되어 있어야 default argument를 활용할 수 있습니다.
* 본 github repository는 FastMRI_challenge 폴더입니다.
* Data 폴더는 제공되며 아래에 상세 구조를 첨부하겠습니다.
* result 폴더는 학습한 모델의 weights을 기록하고 validation, leaderboard dataset의 reconstruction image를 저장하는데 활용합니다. 

### Data 폴더의 구조
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/366c0f08-48c2-4aff-9dba-8f0888ef6902)
* train, val:
    * train, val 폴더는 각각 모델을 학습, 검증하는데 사용하며 각각 image, kspace 폴더로 나뉩니다.
    * generalization과 representation의 trade-off를 잘 고려하여 train, val set을 나누면 됩니다.
    * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: brain_{mask 형식}_{순번}.h5
    * ex) brain_acc8_3.h5  
    * {mask 형식}은 "acc4"과 "acc8" 중 하나입니다.
    * {순번}은 1 ~ 181 사이의 숫자입니다. 
* Leaderboard:
   * Leaderboard는 성능 평가를 위해 활용하는 dataset이므로 절대로 학습 과정에 활용하면 안됩니다.
   * Leaderboard 폴더는 mask 형식에 따라서 acc4과 acc8 폴더로 나뉩니다.
   * acc4과 acc8 폴더는 각각 image, kspace 폴더로 나뉩니다.
   * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: brain_test_{순번}.h5
   * {순번}은 1 ~ 58 사이의 숫자입니다. 

### result 폴더의 구조
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/07923f18-7ea5-4756-ba21-ab9c4bd5d54a)

*

```bash
├── Data
│   ├── Leaderboard
│   │   ├── acc4
│   │   │   ├── image
│   │   │   │   ├── brain_test1.h5
│   │   │   │   ├── brain_test2.h5
│   │   │   │   └── brain_test3.h5
│   │   │   └── kspace
│   │   │       ├── brain_test1.h5
│   │   │       ├── brain_test2.h5
│   │   │       └── brain_test3.h5
│   │   └── acc8
│   │       ├── image
│   │       │   ├── brain_test1.h5
│   │       │   ├── brain_test2.h5
│   │       │   └── brain_test3.h5
│   │       └── kspace
│   │           ├── brain_test1.h5
│   │           ├── brain_test2.h5
│   │           └── brain_test3.h5
│   ├── train
│   │   ├── image
│   │   │   ├── brain_acc4_1.h5
│   │   │   ├── brain_acc4_2.h5
│   │   │   ├── brain_acc4_3.h5
│   │   │   ├── brain_acc4_4.h5
│   │   │   ├── brain_acc4_5.h5
│   │   │   ├── brain_acc8_1.h5
│   │   │   ├── brain_acc8_2.h5
│   │   │   ├── brain_acc8_3.h5
│   │   │   ├── brain_acc8_4.h5
│   │   │   └── brain_acc8_5.h5
│   │   └── kspace
│   │       ├── brain_acc4_1.h5
│   │       ├── brain_acc4_2.h5
│   │       ├── brain_acc4_3.h5
│   │       ├── brain_acc4_4.h5
│   │       ├── brain_acc4_5.h5
│   │       ├── brain_acc8_1.h5
│   │       ├── brain_acc8_2.h5
│   │       ├── brain_acc8_3.h5
│   │       ├── brain_acc8_4.h5
│   │       └── brain_acc8_5.h5
│   └── val
│       ├── image
│       │   ├── brain_acc4_179.h5
│       │   ├── brain_acc4_180.h5
│       │   ├── brain_acc8_180.h5
│       │   └── brain_acc8_181.h5
│       └── kspace
│           ├── brain_acc4_179.h5
│           ├── brain_acc4_180.h5
│           ├── brain_acc8_180.h5
│           └── brain_acc8_181.h5
```

```bash
└── result
    └── test_Unet
        ├── checkpoints
        │   ├── best_model.pt
        │   └── model.pt
        ├── reconstructions_forward
        │   ├── acc4
        │   │   ├── brain_test1.h5
        │   │   ├── brain_test2.h5
        │   │   └── brain_test3.h5
        │   └── acc8
        │       ├── brain_test1.h5
        │       ├── brain_test2.h5
        │       └── brain_test3.h5
        └── reconstructions_val
            ├── brain_acc4_179.h5
            ├── brain_acc4_180.h5
            ├── brain_acc8_180.h5
            └── brain_acc8_181.h5
```

## 2. 폴더 정보
```bash
├── .gitignore
├── evaluate.py
├── leaderboard_eval.py
├── plot.py
├── README.md
├── train.py
└── utils
│   ├── common
│   │   ├── loss_function.py
│   │   └── utils.py
│   ├── data
│   │   ├── load_data.py
│   │   └── transforms.py
│   ├── learning
│   │   ├── test_part.py
│   │   └── train_part.py
│   └── model
│       └── unet.py
├── Data
└── result
```

## How to set?
```bash
conda create -n baby_unet python=3.9
conda activate baby_unet

pip3 install numpy
pip3 install torch
pip3 install h5py
pip3 install scikit-image
pip3 install opencv-python
pip3 install matplotlib
```

## How to train?
```bash
python3 train.py
```

## How to reconstruct
```bash
python3 evaluate.py -m acc4
```

```bash
python3 evaluate.py -m acc8
```

## How to evaluate LeaderBoard Dataset
```bash
python3 leaderboard_eval.py -m acc4
```

```bash
python3 leaderboard_eval.py -m acc8
```

## Let's plot!
```bash
python3 plot.py 
```
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/22dea43d-db54-42c4-9054-1b1ea461c648)
