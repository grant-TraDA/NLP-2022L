import torchmetrics
import torch
import torch.nn.functional as F

import pytorch_lightning as pl

class ModelLightining(pl.LightningModule):
    def __init__(self, model, weights = [1., 1. ,1.], lr = 1e-3):
        super(ModelLightining, self).__init__()
        self.model = model

        metrics = torchmetrics.MetricCollection({
            "accuracy": torchmetrics.Accuracy(num_classes = 3), 
            "microF1": torchmetrics.F1Score(num_classes = 3, average = 'micro'), 
            "macroF1": torchmetrics.F1Score(num_classes = 3, average = 'macro')})

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

        self.f1_per_class = torchmetrics.F1Score(num_classes = 3,average = None)

        self.weights = weights
        self.lr = lr

    def forward(self, bert_tokens, bert_attention, tfidf):
        return self.model(bert_tokens, bert_attention, tfidf)

    def training_step(self, batch, batch_index):
        bert_tokens, bert_attention, tfidf, labels = batch

        output = self.forward(bert_tokens, bert_attention, tfidf)

        loss = F.cross_entropy(output, labels, torch.tensor(self.weights, device=self.device))

        metric_outputs = self.train_metrics(output, labels)
        self.log_dict(metric_outputs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bert_tokens, bert_attention, tfidf, labels = batch

        y_hat = self.forward(bert_tokens, bert_attention, tfidf)

        loss = F.cross_entropy(y_hat, labels, torch.tensor(self.weights, device=self.device))
        metric_outputs = self.valid_metrics(y_hat, labels)
        f1_pc = self.f1_per_class(y_hat, labels)
        self.log_dict(metric_outputs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return f1_pc

    def test_step(self, batch, batch_idx):
        bert_tokens, bert_attention, tfidf, labels = batch
        y_hat = self.forward(bert_tokens, bert_attention, tfidf)
        loss = F.cross_entropy(y_hat, labels, torch.tensor(self.weights, device=self.device))
        self.log("test_loss", loss)
        
    def validation_epoch_end(self, validation_step_outputs):
        all_acc = torch.stack(validation_step_outputs, dim = 0)
        print(f'\n Val F1 per class: {torch.nanmean(all_acc, dim = 0)}')
            

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.lr)