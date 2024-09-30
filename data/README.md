# Data

## 폴더 구성
```plaintext
data
├── raw/          # Raw dataset       
│   └── *         
├── custom/       # Customized dataset       
│   └── *         
├── inference/    # Inference results
│   └── *         
├── analysis/     # Data Analysis
│   └── *         
└── README.md        
```


## train 데이터 구성

### Column
- id : 식별 번호
- source : 데이터 출처
- sentence_1 & sentence_2
- label : 두 문장의 유사도 점수(0~5)
- binary-label: 0 또는 1로 간소화된 점수 <br/>
label 값이 2.5 이상일 때 1, 미만이면 0으로 세팅


### data source
- nsmc : NSMC(Naver Sentiment Movie Corpus)
- petition : 청원 목록
- slack : 슬랙

### data type
* sampled : 샘플링된 데이터
* rtt : round-trip translation