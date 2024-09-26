import re
import emoji
from soynlp.normalizer import repeat_normalize
import torch
import pandas as pd
from torch.utils.data import Dataset
from base import BaseDataLoader
from transformers import ElectraTokenizer, BertTokenizer, BertForMaskedLM

from data_loader.preprocessing import bert_synonym_replacement

# 전처리 함수 정의
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

# BERT 모델과 토크나이저 로드
bert_model_name = 'bert-base-multilingual-cased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForMaskedLM.from_pretrained(bert_model_name)



def clean(text):
    """텍스트 전처리 함수"""
    text = pattern.sub(' ', text)  # 특수문자 제거
    text = emoji.replace_emoji(text, replace='')  # 이모지 제거
    text = url_pattern.sub('', text)  # URL 제거
    text = text.strip()  # 앞뒤 공백 제거
    text = repeat_normalize(text, num_repeats=2)  # 반복 문자 정규화
    return text


class TextDataset(Dataset):
    def __init__(self, sentences_1, sentences_2, labels, tokenizer, max_length, synonym_replacement=False, bert_model=None, bert_tokenizer=None):
        """
        Args:
        - sentences_1: 첫 번째 문장 리스트
        - sentences_2: 두 번째 문장 리스트
        - labels: 라벨 리스트
        - tokenizer: 토크나이저 객체
        - max_length: 최대 시퀀스 길이
        - synonym_replacement: 동의어 대체 적용 여부 (기본값: False)
        - bert_model: BERT 모델 객체 (동의어 대체 시 필요)
        - bert_tokenizer: BERT 토크나이저 객체 (동의어 대체 시 필요)
        """
        if synonym_replacement and bert_model is not None and bert_tokenizer is not None:
            # 동의어 대체를 적용하여 텍스트 전처리
            self.sentences_1 = [bert_synonym_replacement(clean(sentence), bert_model, bert_tokenizer) for sentence in sentences_1]
            self.sentences_2 = [bert_synonym_replacement(clean(sentence), bert_model, bert_tokenizer) for sentence in sentences_2]
        else:
            # 동의어 대체 없이 텍스트 전처리
            self.sentences_1 = [clean(sentence) for sentence in sentences_1]
            self.sentences_2 = [clean(sentence) for sentence in sentences_2]

        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences_1)

    def __getitem__(self, idx):
        sentence_1 = self.sentences_1[idx]
        sentence_2 = self.sentences_2[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence_1,
            sentence_2,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TextDataLoader(BaseDataLoader):
    """
    Text data loading demo using BaseDataLoader with Synonym Replacement
    """

    def __init__(self, config, batch_size=32, shuffle=True, validation_split=0.0, num_workers=1,
                 synonym_replacement=True):
        # CSV 파일 읽기
        data = pd.read_csv(config['data']['train_file'])

        # 텍스트와 라벨 추출
        sentences_1 = data['sentence_1'].tolist()
        sentences_2 = data['sentence_2'].tolist()
        labels = data['label'].tolist()

        # 전처리 및 토크나이저 설정
        tokenizer_name = config['arch']['args']['model_name']
        max_length = config['max_length']

        # 토크나이저 및 모델 로드
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)
        bert_model_name = 'bert-base-multilingual-cased'
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_model = BertForMaskedLM.from_pretrained(bert_model_name)

        # 전처리된 데이터셋 생성
        dataset = TextDataset(
            sentences_1,
            sentences_2,
            labels,
            tokenizer,
            max_length,
            synonym_replacement=synonym_replacement,
            bert_model=bert_model,
            bert_tokenizer=bert_tokenizer
        )

        # 데이터로더 생성
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

