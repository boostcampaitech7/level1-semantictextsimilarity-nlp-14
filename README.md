# Semantic Text Similarity Baseline

## Structure

https://github.com/boostcampaitech5/level1_semantictextsimilarity-nlp-06 <br/>
전체 구조는 해당 깃허브를 참고하여 만들었습니다

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

## Description
baseline 활용 예시입니다

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

1. inference.py의 line 40에서 불러올 모델 세팅합니다

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