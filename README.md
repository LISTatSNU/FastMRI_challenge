# 2026 baby unet (annotation)
2026 SNU FastMRI challenge — knee annotation track

* 본 대회는 **knee 데이터만** 사용합니다.
* 복원 품질은 두 가지 SSIM으로 평가합니다.
  * **SSIM_full**: foreground mask 내부 영역에 대해 평균한 SSIM
  * **SSIM_bbox**: fastMRI+ 병변 bounding-box 영역 안에서만 계산한 SSIM

## 1. 폴더 계층

### 폴더의 전체 구조
![image](docs/fastmri_folder_structure.png)
* `FastMRI_challenge`, `Data`, `result` 폴더가 위의 구조대로 설정되어 있어야 default argument를 활용할 수 있습니다.
* 본 github repository는 `FastMRI_challenge` 폴더입니다.
* `Data` 폴더는 MRI data 파일을 담고 있으며 아래에 상세 구조를 첨부하겠습니다.
* `result` 폴더는 학습한 모델의 weights을 기록하고 validation dataset의 reconstruction image를 저장하는데 활용되며, 아래에 상세 구조를 첨부하겠습니다.

### Data 폴더의 구조
![image](docs/fastmri_data_structure.png)
* train, val:
    * train, val 폴더는 각각 모델을 학습(train), 검증(validation)하는데 사용하며 각각 image, kspace 폴더로 나뉩니다.
    * 참가자들은 generalization과 representation의 trade-off를 고려하여 train, validation의 set을 자유로이 나눌 수 있습니다.
    * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: knee\_{mask 형식}\_{순번}.h5
    * ex) knee_acc4_10.h5, knee_acc8_3.h5
    * {mask 형식}은 "acc4", "acc8" 중 하나입니다.
    * train은 "acc4", "acc8" 각각 {순번} 1 ~ 85, val은 각각 1 ~ 15 입니다.

> phase1에서는 train, val 데이터만 제공됩니다. 성능 평가용 leaderboard 데이터셋은 추후 별도로 배포될 예정입니다.

### Annotation
* fastMRI+ knee 병변 bounding box는 각 image h5의 `attrs['annotations']`에 JSON으로 저장되어 있습니다.
* 형식: `{ "<slice>": [ {"x", "y", "width", "height", "label"}, ... ] }` (384 x 384 image 공간 기준, width·height가 16 초과인 박스만 포함)

### result 폴더의 구조
* result 폴더는 모델의 이름에 따라서 여러 폴더로 나뉠 수 있습니다.
* test_Unet (혹은 test_Varnet) 폴더는 아래 항목들로 구성되어 있습니다.
  * checkpoints - `model.pt`, `best_model.pt`의 정보가 있습니다. 모델의 weights 정보를 담고 있습니다.
  * reconstructions_val - validation dataset의 reconstruction을 저장합니다. knee\_{mask 형식}\_{순번}.h5 형식입니다. (```train.py``` 참고)
  * val_loss_log.npy - epoch별로 validation loss를 기록합니다. (```train.py``` 참고)

## 2. 폴더 정보

```bash
├── .gitignore
├── README.md
├── reconstruct.py
├── requirements.txt
├── train.py
├── tutorial.ipynb
└── utils
│   ├── common
│   │   ├── loss_function.py
│   │   ├── metrics.py
│   │   └── utils.py
│   ├── data
│   │   ├── load_data.py
│   │   └── transforms.py
│   ├── learning
│   │   ├── test_part.py
│   │   └── train_part.py
│   └── model
│       └── unet.py
└── result
```

## 3. Before you start
* phase1에서는 ```train.py```로 학습/검증을 진행합니다.
* ```train.py```
   * train/validation을 진행하고 학습한 model의 결과를 result 폴더에 저장합니다.
   * 가장 성능이 좋은 모델의 weights을 ```best_model.pt```으로 저장합니다.
* val 자가채점
   * ```utils/common/metrics.py```의 ```ssim_full```, ```ssim_bbox```로 본인 validation reconstruction의 SSIM을 직접 계산할 수 있습니다. (annotation은 각 image h5의 ```attrs['annotations']```에 포함)
* ```reconstruct.py``` (leaderboard 데이터 배포 후 사용)
   * ```train.py```으로 학습한 ```best_model.pt```을 활용해 leaderboard dataset을 reconstruction하는 스크립트입니다.
   * phase1에서는 leaderboard 데이터가 제공되지 않으므로 아직 실행할 수 없습니다. 추후 leaderboard 데이터가 배포되면 그때 의미 있게 사용하시면 됩니다. (미리 코드 구조를 살펴보는 것은 자유입니다.)

## 4. How to set?
(python 3.12.9)
```bash
pip3 install -r requirements.txt
```

## 5. How to train?
```bash
python train.py // sh train.sh
```
- validation할 때, reconstruction data를 ```result/reconstructions_val/```에 저장합니다.
- epoch 별로 validation dataset에 대한 loss를 기록합니다.
- sh train.sh를 사용하여도 같은 결과를 얻으실 수 있습니다. Hyperparameter를 쉽게 조작할 수 있습니다.
- **seed 고정**을 하여 이후에 Re-training하였을 때 **같은 결과가 나와야 합니다**.

## 6. How to reconstruct? (leaderboard 데이터 배포 후)
> phase1에서는 leaderboard 데이터가 제공되지 않으므로 이 단계는 아직 실행할 수 없습니다. leaderboard 데이터가 배포된 이후 아래 절차로 진행하시면 됩니다.

```bash
python reconstruct.py // sh reconstruct.sh
```
- leaderboard 평가를 위한 reconstruction data를 ```result/reconstructions_leaderboard```에 저장합니다.
- acc4 / acc8 각각의 reconstruction 시간과 총합을 함께 출력합니다.

## 7. What to submit!
- github repository(코드 실행 방법 readme에 상세 기록)
- loss 그래프 혹은 기록
- 모델 weight file
- 모델 설명 ppt
