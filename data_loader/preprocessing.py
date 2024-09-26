from transformers import BertForMaskedLM, BertTokenizer
import torch
import random


def bert_synonym_replacement(sentence, model, tokenizer, mask_token='[MASK]', top_k=5):
    """
    BERT를 사용하여 문장의 동의어 대체를 수행

    Args:
    - sentence (str): 원본 문장
    - model (transformers.BertForMaskedLM): 사전 학습된 BERT 모델
    - tokenizer (transformers.BertTokenizer): BERT 토크나이저
    - mask_token (str): BERT 마스크 토큰 (기본값: [MASK])
    - top_k (int): 대체 후보 단어의 개수 (기본값: 5)

    Returns:
    - str: 동의어 대체가 적용된 문장
    """
    words = sentence.split()
    if len(words) < 3:
        return sentence

    # 무작위로 단어 하나 선택
    random_idx = random.choice(range(1, len(words) - 1))
    target_word = words[random_idx]

    # 선택된 단어를 [MASK]로 대체
    masked_sentence = ' '.join(words[:random_idx] + [mask_token] + words[random_idx + 1:])
    inputs = tokenizer(masked_sentence, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    # [MASK] 위치의 단어 예측
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    mask_token_logits = outputs.logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

    # 랜덤하게 대체 단어 선택
    synonym = tokenizer.decode(random.choice(top_k_tokens)).strip()

    # 대체된 문장 반환
    new_sentence = ' '.join(words[:random_idx] + [synonym] + words[random_idx + 1:])
    return new_sentence
