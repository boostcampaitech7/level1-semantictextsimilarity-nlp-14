import torch.nn.functional as F
from base import BaseModel
from transformers import ElectraTokenizer, ElectraForPreTraining


class KcELECTRAModel(BaseModel):
    def __init__(self, config):
        super(KcELECTRAModel, self).__init__()
        model_name = config['arch']['args']['model_name']

        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraForPreTraining.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def get_loss(self, logits, labels):
        return F.cross_entropy(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

