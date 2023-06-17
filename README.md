# 2023 baby unuet
2023 SNU FastMRI challenge

## Folder

```bash
├── Dataset
│   ├── Leaderboard
│   │   ├── acc4
│   │   │   ├── image
│   │   │   │   ├── brain_test1.h5
│   │   │   │   ├── brain_test2.h5
│   │   │   │   └── ...
│   │   │   └── kspace
│   │   │       ├── brain_test1.h5
│   │   │       ├── brain_test2.h5
│   │   │       └── ...
│   │   └── acc8
│   │       ├── image
│   │       │   ├── brain_test1.h5
│   │       │   ├── brain_test2.h5
│   │       │   └── ...
│   │       └── kspace
│   │           ├── brain_test1.h5
│   │           ├── brain_test2.h5
│   │           └── ...
│   ├── Training
│   │   ├── image
│   │   │   ├── brain_acc4_1.h5
│   │   │   ├── brain_acc4_2.h5
│   │   │   ├── ...
│   │   │   ├── brain_acc8_1.h5
│   │   │   ├── brain_acc8_2.h5
│   │   │   └── ...
│   │   └── kspace
│   │       ├── brain_acc4_1.h5
│   │       ├── brain_acc4_2.h5
│   │       ├── ...
│   │       ├── brain_acc8_1.h5
│   │       ├── brain_acc8_2.h5
│   │       └── ...
│   └── Validation
│       ├── image
│       │   ├── ...
│       │   ├── brain_acc4_181.h5
│       │   ├── ...
│       │   └── brain_acc8_181.h5
│       └── kspace
│           ├── ...
│           ├── brain_acc4_181.h5
│           ├── ...
│           └── brain_acc8_181.h5
│
├── evaluate.py
├── leaderboard_eval.py
├── plot.ipynb
├── plot.py
├── README.md
├── train.ipynb
├── train.py
└── utils
    ├── common
    │   ├── loss_function.py
    │   └── utils.py
    ├── data
    │   ├── load_data.py
    │   └── transforms.py
    ├── learning
    │   ├── test_part.py
    │   └── train_part.py
    └── model
        └── unet.py
```
## Requirements
```bash
conda create -n baby_unet
conda activate baby_unet
pip3 install torch
pip3 install scipy
```

## Train
```bash
python3 train.py -g 0 -b 4 -e 10 -l 0.003 -r 1 -n baby_unet -t ./Dataset/Training/image/ -v ./Dataset/Validation/image/
```

## Evaluate(Evaluation set)

## Evaluate(LeaderBoard Dataset)

## Plot
