import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, text_df, embedder, todense=False):
        self.embedder = embedder
        self.text_df = text_df
        self.todense = todense
        self.dp = {}

    def __len__(self):
        return len(self.text_df)

    def __getitem__(self, i):
        data = self.text_df['sentence'][i]
        label = self.text_df['target'][i]
        label_tensor = np.array([0] * 3)
        label_tensor[int(label)] = 1
        label_tensor = torch.from_numpy(label_tensor.astype(np.float32)).flatten()
        if i in self.dp:
            predictors = self.dp[i]
        else:
            if self.embedder.todense:
                predictors = torch.from_numpy(self.embedder.transform([data]).todense().astype(np.float32)).flatten()
            else:
                predictors = torch.from_numpy(self.embedder.transform([data]).astype(np.float32)).flatten()
            self.dp[i] = predictors
        return predictors, label_tensor
