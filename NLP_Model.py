import tez
import torch
import torch.nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
import pandas as pd

class BERTDataset: 
    def __init__(self, texts, targets, max_len = 64):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", 
                                                                    do_lower_case=True)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.endoce_plus(text, None,
                                            add_special_tokens = True,
                                            mex_length = self.max_len,
                                            padding = 'max_length',
                                            truncation = True)

        resp = {
            'ids': torch.tensor(inputs['input_ids'],
                                dtype = torch.long),

            'mask': torch.tensor(inputs['attention_mask'], 
                                dtype = torch.long),

            'token_type_ids': torch.tensor(inputs['token_type_ids'],
                                           dtype = torch.long),

            'targets': torch.tensor(self.targets[idx],
                                           dtype = torch.long)
        } 

        return resp


class TextModel(tex.Model):
    def __init__(self, num_classes, num_train_steps):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", 
                                                            return_dict = False)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = 'batch'
    
    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr = 3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(self.optimizer, 
                                              num_warmup_steps = 0,
                                              num_training_steps = self.num_train_steps)

        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

    def monitor_metrics(self, outputs, targets):
        outputs = outputs.cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()

        return {'accuracy': metrics.accuracy_score(targets, outputs)}

    def forward(self, ids, mask, token_type_ids, targets = None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids = token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x,targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met
        return x, 0, {}

def train_model(fold):
    df = pd.read_csv()