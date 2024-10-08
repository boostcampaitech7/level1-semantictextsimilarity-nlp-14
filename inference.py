## 작성 중

import argparse
import yaml
import pandas as pd
import os
from tqdm.auto import tqdm

import torch

# import transformers
# import pandas as pd

import pytorch_lightning as pl

# import wandb
##############################
from utils import data_pipeline, tools


if __name__ == "__main__":
    # dataset path 설정
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="./data/raw/train.csv")
    parser.add_argument("--dev_path", default="./data/raw/dev.csv")
    parser.add_argument("--test_path", default="./data/raw/dev.csv")
    ## test_path도 dev.csv로 설정 >> trainer.test()에서 dev.csv 사용
    parser.add_argument("--predict_path", default="./data/raw/test.csv")
    ## predict_path에 test.csv로 설정 >> trainer.predict()에서 test.csv 사용
    args = parser.parse_args(args=[])

    # baseline_config 설정 불러오기
    with open("baselines/baseline_config.yaml", encoding="utf-8") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # inference에 쓸 모델 불러오기(CFG로 참조)
    model_path = CFG["inference"]["model_path"]

    # dataloader / model 설정
    seed_num = CFG["seed"]
    init_seed = tools.init_seed(seed_num)
    dataloader = data_pipeline.Dataloader(
        CFG, args.train_path, args.dev_path, args.test_path, args.predict_path
    )
    model = torch.load(model_path)

    early_stopping_callbacks = pl.callbacks.EarlyStopping(
        monitor=CFG["early_stopping"]["monitor"],
        patience=CFG["early_stopping"]["patience"],
        mode=CFG["early_stopping"]["mode"],
    )

    # trainer 인스턴스 생성
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[early_stopping_callbacks],
        max_epochs=CFG["train"]["epoch"],
        log_every_n_steps=1,
    )

    # Inference part
    predictions = trainer.predict(model=model, datamodule=dataloader)
    ## datamodule에서 predict_dataloader 호출

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("./data/raw/sample_submission.csv")
    output["target"] = predictions
    output.to_csv("./data/inference/output.csv", index=False)
