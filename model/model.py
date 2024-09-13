import torch
import transformers
import torchmetrics
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = CFG['train']['model_name']
        self.lr = CFG['train']['LR']
        ## configure_optimizers에서 사용
        
        ## CFG에 설정된 lossF와 optim을 문자열로 저장
        loss_choice  = CFG['train']['LossF']
        optim_choice = CFG['train']['optim']

        ## CFG의 model_name으로 설정된 모델 불러오기
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=1)
        
        ## 현서님 의견
        ## ForSequenceClassification 쓰면 num_labels=1로 linear layer 추가
        ## 그냥 AutoModel 뒤에 우리가 원하는 FC layer 추가해보면 더 성능 좋게 나오지 않을까?
        
        ## ipynb로 cell 단위로 실행하면 model 불러와서 자세한 구조 확인할 수 있다

        ## 재룡님 의견
        ## pre-trained가 어느 데이터에 대해 사전훈련된 건지 확인해보고
        ## 우리가 쓰는 데이터셋과 비슷하다면 분류기를 재정의하는 방향으로 해볼 수 있을것 같다

        
        ## 문자열로 표현된 loss와 optimizer를 함수로 변환
        self.loss_func = eval(loss_choice)()
        self.optim = eval(optim_choice)
        ## self.optim은 configure_optimizers에서 사용

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()


    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr)
        return optimizer
