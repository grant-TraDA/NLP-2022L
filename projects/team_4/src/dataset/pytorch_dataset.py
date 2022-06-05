import torch
from torch.utils.data import Dataset
import numpy as np

class HateSpeechDataset(Dataset):
    def __init__(self, data_df, bert_tokenizer, tfidf_tokenizer):
        self.data_df = data_df
        self.bert_tokenizer = bert_tokenizer
        self.tfidf_tokenizer = tfidf_tokenizer

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        tf_emotes = torch.from_numpy(self.tfidf_tokenizer.transform([self.data_df['Text'][idx]]).todense().astype(np.float32)).flatten()
        label = self.data_df['Tag'][idx]
        tokenizer_output = self.bert_tokenizer([self.data_df['Text'][idx]], return_tensors = "pt", padding='max_length', max_length = 200, truncation = True) #is_split_into_words=True
        return tokenizer_output['input_ids'].flatten(), tokenizer_output['attention_mask'].flatten(), tf_emotes, torch.tensor(label).long() #907