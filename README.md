# 2023 baby unet
2023 SNU FastMRI challenge

## Folder

```bash
├── .gitignore
├── evaluate.py
├── leaderboard_eval.py
├── plot.ipynb
├── plot.py
├── README.md
├── train.ipynb
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
    ├── baby_unet
    │   ├── checkpoints
    │   │   ├── best_model.pt
    │   │   └── model.pt
    │   ├── reconstructions_forward
    │   │   ├── acc4
    │   │   │   ├── brain_test1.h5
    │   │   │   ├── brain_test2.h5
    │   │   │   └── brain_test3.h5
    │   │   └── acc8
    │   │       ├── brain_test1.h5
    │   │       ├── brain_test2.h5
    │   │       └── brain_test3.h5
    │   └── reconstructions_val
    │       ├── brain_acc4_179.h5
    │       ├── brain_acc4_180.h5
    │       ├── brain_acc8_180.h5
    │       └── brain_acc8_181.h5
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


## Requirements
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

## Train
```bash
python3 train.py -g 0 -b 1 -e 10 -l 0.003 -r 1
```

## Reconstruction
```bash
python3 evaluate.py -g 0 -b 1 -m acc4
```

```bash
python3 evaluate.py -g 0 -b 1 -m acc8
```

## Evaluate(LeaderBoard Dataset)
```bash
python3 leaderboard_eval.py -g 0 -m acc4
```

```bash
python3 leaderboard_eval.py -g 0 -m acc8
```

## Plot
```bash
python3 plot.py 
```
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/22dea43d-db54-42c4-9054-1b1ea461c648)
