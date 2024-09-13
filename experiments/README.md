# experiments

## 정보
- 실험 환경마다 별도의 폴더를 만들어서 학습된 모델 저장

- 폴더 이름은 실험 날짜와 시간, 실험자 이름으로 구성
```plaintex
9월 12일 14시에 admin : gildong 으로 train.py 실행하면
./experiments/09-12_14_gildong 폴더에 model.pt 저장
```
<br/>

## 실험 기록
- 실험할 때 세팅한 config 정보와 실험 결과를 기록합니다.

- 자세한 내용은 ./lightning_logs/version_n 폴더에 있습니다. <br/>
---

### 09-11_15_eyeol
- 실험 config : <br/>
```plaintext
model_name  :  KO-BERT
epoch       :  1
batch_size  :  32
```
<br>

- loss와 pearson 점수 : <br/>
```plaintext
val_loss    :  2.5086
val_pearson :  0.1606
```
./lightning_logs/version_3 참고

---

### 09-11_15_eyeol
- 실험 config : <br/>
```plaintext
model_name  :  KO-BERT
epoch       :  20
batch_size  :  32
```
<br>

- loss와 pearson 점수 : <br/>
```plaintext
val_loss    :  2.1219
val_pearson :  0.3201
```
./lightning_logs/version_4 참고

---

### 09-11_18_eyeol

- 실험 config : <br/>
```plaintext
model_name  :  KR-ELECTRA
epoch       :  10
batch_size  :  32
```
<br>

- loss와 pearson 점수 : <br/>
```plaintext
val_loss    :  0.6424
val_pearson :  0.8470
```
./lightning_logs/version_5 참고

---

### 09-11_20-eyeol
- 실험 config : <br/>
```plaintext
model_name  :  KR-ELECTRA
epoch       :  20
batch_size  :  32
```
<br>

- loss와 pearson 점수 : <br/>
```plaintext
val_loss    :  0.6281
val_pearson :  0.8517
```
./lightning_logs/version_6 참고

---

### 09-12_09-eyeol

- 실험 config : <br/>
```plaintext
model_name  :  KR-ELECTRA
epoch       :  20
batch_size  :  16   # batch size 변경
```
<br>

- loss와 pearson 점수 : <br/>
```plaintext
val_loss    :  0.5541
val_pearson :  0.8789
```

./lightning_logs/version_10 참고

---