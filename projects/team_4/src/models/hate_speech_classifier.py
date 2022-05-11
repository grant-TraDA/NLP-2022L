import torch
from torch import nn

class HateSpeechClassifier(nn.Module):
    def __init__(self, bert_model, classification_model, train_bert = False):
        super().__init__()
        self.bert = bert_model

        if not train_bert:
          for param in self.bert.parameters():
            param.requires_grad = False

        self.classification_model = classification_model


    def forward(self, bert_tokens_ids, bert_tokens_attention, tfidf_features):
      _, cls_embeds = self.bert(input_ids = bert_tokens_ids, attention_mask = bert_tokens_attention, return_dict=False)
      features_connected = torch.cat((cls_embeds, tfidf_features), 1)

      return self.classification_model(features_connected)