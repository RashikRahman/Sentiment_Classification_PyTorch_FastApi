from fastapi import FastAPI
from pydantic.types import PositiveFloat
import tez
import torch
import torch.nn as nn
import transformers
from pydantic import BaseModel

app = FastAPI()


class SentimentPredict(BaseModel):
    text: str
    threshold: float


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
    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.bert = transformers.BertModel.from_pretrained(
            "bert-base-uncased", return_dict=False
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1) # 768,1 we write 1 as it is a binary classification

    def forward(self, ids, mask, token_type_ids, targets = None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids = token_type_ids)
        x = self.bert_drop(x)
        x = torch.sigmoid(self.out(x))
        return x, 0, {}

Model = TextModel()
Model.load('logs/model.bin', device='cuda')

@app.get("/")
def read_root():
    return {"Hello": "World"}

    
@app.get("/predict")
def fetch_predictions(text: str):
    data = BERTDataset([text], [-1])
    prediction = float(list(Model.predict(data, batch_size=1))[0][0][0])
    if prediction>0.95:
        Sentiment = 'Positive'
    else:
        Sentiment = 'Negative'
    return {"Positive": prediction,
            "Negative": 1-prediction,
            "FeedBack": text,
            "Sentiment": Sentiment}