import transformers
import torchmetrics
import torch  # eval로 torch import하는 부분에서 필요
import pytorch_lightning as pl
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup


class Model(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = CFG["train"]["model_name"]
        self.lr = CFG["train"]["LR"]
        # configure_optimizers에서 사용
        self.weight_decay = CFG["train"]["weight_decay"]
        self.num_hiddens = CFG["train"]["num_hiddens"]
        self.num_warmup_rate = CFG["LR_scheduler"]["num_warmup_rate"]

        self.step = CFG["LR_scheduler"]["LR_step_type"]
        self.freq = CFG["LR_scheduler"]["LR_step_freq"]

        # CFG에 설정된 lossF와 optim을 문자열로 저장
        loss_choice = CFG["train"]["LossF"]
        optim_choice = CFG["train"]["optim"]

        # CFG의 model_name으로 설정된 모델 불러오기
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=self.num_hiddens
        )

        # 단순한 선형 레이어를 추가해 모델의 학습과정에서 비선형성을 추가로 배울 수 있도록 함.
        if self.num_hiddens != 1:
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(CFG["train"]["dropout"])
            self.linear = nn.Linear(self.num_hiddens, 1)

        # 현서님 의견
        # ForSequenceClassification 쓰면 num_labels=1로 linear layer 추가
        # 그냥 AutoModel 뒤에 우리가 원하는 FC layer 추가해보면 더 성능 좋게 나오지 않을까?

        # ipynb로 cell 단위로 실행하면 model 불러와서 자세한 구조 확인할 수 있다

        # 재룡님 의견
        # pre-trained가 어느 데이터에 대해 사전훈련된 건지 확인해보고
        # 우리가 쓰는 데이터셋과 비슷하다면 분류기를 재정의하는 방향으로 해볼 수 있을것 같다

        # 문자열로 표현된 loss와 optimizer를 함수로 변환
        self.loss_func = eval(loss_choice)()
        self.optim = eval(optim_choice)
        # self.optim은 configure_optimizers에서 사용

    def forward(self, input):
        input_ids = input["input_ids"]
        attention_mask = input.get("attention_mask", None)
        token_type_ids = input.get("token_type_ids", None)
        x = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )["logits"]

        if x.size(-1) == 1:
            return x
        else:  # 비교적 덜 복잡한 electra 모델에서는 효율 떨어짐
            x = self.gelu(x)
            x = self.dropout(x)
            x = self.linear(x)
            return x

    def training_step(self, batch, batch_idx):
        y = batch["targets"]
        logits = self(input=batch)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["targets"]
        logits = self(input=batch)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        y = batch["targets"]
        logits = self(input=batch)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        logits = self(input=batch)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = self.optim(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(self.num_warmup_rate * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.step,  # 매 배치마다 업데이트
            "frequency": self.freq,  # 매 interval마다
        }

        return [optimizer], [scheduler_config]
