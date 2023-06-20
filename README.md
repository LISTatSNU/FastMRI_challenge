# 2023 baby varnet
2023 SNU FastMRI challenge

## Folder

```bash
├── evaluate.py
├── leaderboard_eval.py
├── train.py
└── utils
│    ├── common
│    │   ├── loss_function.py
│    │   └── utils.py
│    ├── data
│    │   ├── load_data.py
│    │   └── transforms.py
│    ├── learning
│    │   ├── test_part.py
│    │   └── train_part.py
│    └── model
│        ├── fastmri
│        │   ├── coil_combine.py
│        │   ├── data
│        │   │   ├── __init__.py
│        │   │   ├── mri_data.py
│        │   │   ├── README.md
│        │   │   ├── subsample.py
│        │   │   ├── transforms.py
│        │   │   └── volume_sampler.py
│        │   ├── fftc.py
│        │   ├── __init__.py
│        │   ├── losses.py
│        │   ├── math.py
│        ├── unet.py
│        └── varnet.py
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
TODO
conda create -n baby_varnet python=3.9
conda activate baby_varnet

pip3 install torch
pip3 install numpy
pip3 install requests
pip3 install tqdm
pip3 install h5py
pip install scikit-image
pip install pyyaml
pip3 install opencv-python
```

## Train
```bash
python3 train.py
```

## Reconstruction
```bash
python3 evaluate.py -m acc4
```

```bash
python3 evaluate.py -m acc8
```

## Evaluate(LeaderBoard Dataset)
```bash
python3 leaderboard_eval.py -m acc4
```

```bash
python3 leaderboard_eval.py -m acc8
```

## Plot (당장은 없음)
```bash
python3 plot.py 
```
![image](https://github.com/LISTatSNU/FastMRI_challenge/assets/39179946/22dea43d-db54-42c4-9054-1b1ea461c648)

