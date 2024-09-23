from tqdm import tqdm

import torch
import transformers
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)
    


# 나중에 Custom 가능
class Dataloader(pl.LightningDataModule):
    # instance 생성할 때 CFG(baseline_config 세팅) 입력
    def __init__(self, CFG, train_path, dev_path, test_path, predict_path):
        super().__init__()
        # config
        self.model_name = CFG['train']['model_name']
        self.batch_size = CFG['train']['batch_size']
        self.shuffle = CFG['train']['shuffle']

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        
    # tokenizing과 preprocessing은 나중에 Custom 가능
    def tokenizing(self, dataframe):
        # 어텐션 마스크 추가 안했으나, 추가시 성능 높아질 확률 높음
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
        # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.

            text = [item[text_column] for text_column in self.text_columns]
            outputs = self.tokenizer(
                text[0],
                text[1],
                add_special_tokens=True,
                padding='max_length',  # max_length로 패딩을 고정
                truncation=True,       # 텍스트를 최대 길이로 자름
                max_length=160         # max_length 설정
            )
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            
            # swap augmentation
            inv_train = train_data.iloc[:, [0, 1, 3, 2, 4, 5]]
            inv_train.columns = train_data.columns
            train_data = pd.concat([train_data,inv_train], ignore_index=True)

            # 단순히 stack한거라 shuffle을 true로 하지 않으면 편향 생김
            # seed 고정 안하고 훈련시 shuffle 바뀌어서 성능이 변경되어버림
            train_data = pd.concat([train_data,train_data],ignore_index=True)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)
