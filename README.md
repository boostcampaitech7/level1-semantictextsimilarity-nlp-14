# Semantic Text Similarity Project

STS 문장 유사도 측정을 위한 프로젝트입니다


## Description
- 프로젝트 기간 : 2024.09.10(화) ~ 2024.09.26(목)
- 데이터셋
  * train set : 9,324개
  * dev set   : 550개
  * test set  : 1,100개
- 평가방법 : 피어슨 상관계수

> 네이버 부스트캠프 AI Tech 7기 Level 1 프로젝트


## How to Start

### 1. Clone Repository

```sh
$ git clone https://github.com/boostcampaitech7/level1-semantictextsimilarity-nlp-14.git
$ cd level1-semantictextsimilarity-nlp-14
```

### 2. Copy Config File

```sh 
$ cp baselines/baseline_config.yaml.example baselines/baseline_config.yaml
```

### 3. Set Your Experiment Config
`baseline_config.yaml`에서 원하는 세팅으로 수정하시면 됩니다

```yaml
admin: admin                                     # your name
seed: 42                                         
train:
  model_name: snunlp/KR-ELECTRA-discriminator    # model
  batch_size: 32
  epoch: 1
  LR: 0.00003
  LossF: torch.nn.MSELoss 
  optim: torch.optim.AdamW
  ## LossF와 optim은 torch.nn과 torch.optim을 꼭 적어야 합니다
  weight_decay: 0.01
  num_hiddens: 1
  dropout: 0.1
  num_workers: 7                                 # 사용하는 cpu core 숫자
LR_scheduler:
  num_warmup_rate: 0.1
  LR_step_type: step 
  LR_step_freq: 1
early_stopping:
  monitor: val_loss
  patience: 3
  mode: min
inference:
  model_path: ./experiments/09-12_16_eyeol/model.pt    # this works in inference.py
```

### 4. Prepare Dataset
프로젝트를 시작하기 위해, 데이터를 `data/raw` 폴더에 넣어주세요.

이 폴더는 원본 데이터를 저장하는 곳입니다.
```plaintext
data/raw/train.csv
data/raw/dev.csv
data/raw/test.csv
```

### 5. Set python virtual environment
root 폴더에서 다음 명령어를 실행
```sh
$ conda env create -f environment.yaml
```

생성한 가상환경 실행
```sh
$ conda activate sts
(sts) $                # 프롬프트 왼쪽에 (sts)가 생기면 성공
```

### 6. train.py 실행
모델 학습을 위해 train.py를 실행합니다
```sh
(sts) $ python train.py
```

학습이 끝나면, experiments 폴더에 학습된 모델이 저장됩니다


### 7. inference.py 실행
baseline_config.yaml에서 inference.model_path를
학습된 모델의 경로로 수정해주세요

`baselines/baseline_config.yaml`:
```yaml
inference:
  model_path: ./experiments/09-12_16_eyeol/model.pt
```

inference.py를 실행하면, 학습된 모델을 불러와서 output.csv를 출력합니다
```sh
(sts) $ python inference.py
```
output.csv는 `./data/inference` 폴더에 저장됩니다


## Structure

```plaintext
STS
├── data/                  # input/output
├── model/       
│   └── model.py           # model 정의
├── utils/    
│   ├── data_pipeline.py   # Dataloader
│   └── utils.py            
├── train.py               # 모델 학습 및 저장
├── inference.py           # output.csv 출력
│
├── baselines/             # 실험 세팅
│   └── baseline_config.yml 
├── experiments/           # 실험마다 모델 저장
├── lightning_logs/        # 자세한 실험 내역
└── README.md        

```

> https://github.com/boostcampaitech5/level1_semantictextsimilarity-nlp-06
>
> 이전 기수의 baseline code를 참고해서 만들었습니다

## 활용 예시


```plaintext
## 훈련 ##

0. baseline_config.yaml 원하는 세팅으로 수정

1. train.py 실행하여 모델을 학습시킵니다
    ├─ data_pipeline.py: Dataloader class 제공
    └─ model.py : Model class 제공
    
2. experiments 폴더에 학습된 모델이 저장됩니다

3. 실험 종료 후 val_pearson 값을 확인하고
   experiments/README.md에 결과를 기록합니다


## 예측 ## 

1. inference.py의 37 line에서 불러올 모델 세팅합니다

2. inference.py 실행하여 output.csv를 출력합니다

3. 리더보드에 제출 후 결과 분석 시작


## 분석 ## 

1. 자기만의 또는 팀만의 스타일로 분석
ex) Binary-label로 Confusion Matrix 만들기
    특별히 정확도가 떨어지는 label 값이 있나 확인

2. 분석 결과에 따라 모델 피드백
ex) tokenizer 교체 / hyperparameter 수정
    데이터 증강 / 특정 벤치마크셋으로 사전학습 추가
    다른 모델로 교체 / low-level에서 모델 커스텀

3. 다시 훈련 후 평가
```

## Collaborators

<h3 align="center">NLP-14조 Word Maestro(s)</h3>

<div align="center">

|          [김현서](https://github.com/kimhyeonseo0830)          |          [단이열](https://github.com/eyeol)          |          [안혜준](https://github.com/jagaldol)          |          [이재룡](https://github.com/So1pi)          |          [장요한](https://github.com/DDUKDAE)          |
| :----------------------------------------------------: | :-----------------------------------------------------: | :------------------------------------------------------: | :---------------------------------------------------: | :---------------------------------------------------: |
| <img src="https://github.com/kimhyeonseo0830.png" width="100"> | <img src="https://github.com/eyeol.png" width="100"> | <img src="https://github.com/jagaldol.png" width="100"> | <img src="https://github.com/So1pi.png" width="100"> | <img src="https://github.com/DDUKDAE.png" width="100"> |

</div>
