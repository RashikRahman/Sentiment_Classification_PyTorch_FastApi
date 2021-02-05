import pandas as pd
import tez
import torch
import torch.nn as nn
import transformers
from sklearn import metrics, model_selection
from transformers import AdamW, get_linear_schedule_with_warmup


class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = 64

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }


class TextModel(tez.Model):
    def __init__(self, num_train_steps):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1) # 768,1 we write 1 as it is a binary classification
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr = 3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def forward(self, ids, mask, token_type_ids, targets = None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids = token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)
        if targets is not None:
            loss = self.loss(x,targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met
        return x, 0, {}

if __name__ == "__main__":
    dfx = pd.read_csv(r"G:\DS\3.Personal_Projects\Sentiment_Classification_PyTorch_FastApi\Data\imdb.csv")
    dfx.sentiment = dfx.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = BERTDataset(
        review=df_train.review.values, target=df_train.sentiment.values
    )

    valid_dataset = BERTDataset(
        review=df_valid.review.values, target=df_valid.sentiment.values
    )

    n_train_steps = int(len(df_train) / 32 * 10)
    model = TextModel(num_train_steps=n_train_steps)


    tb_logger = tez.callbacks.TensorBoardLogger(log_dir="logs/")
    es = tez.callbacks.EarlyStopping(monitor="valid_loss", model_path="model.bin")
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=64,
        device="cuda",
        epochs=50,
        callbacks=[tb_logger, es],
        fp16=True,
    )
    model.save("logs/model.bin")