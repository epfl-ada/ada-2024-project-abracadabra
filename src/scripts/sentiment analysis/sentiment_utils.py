import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

device = "mps"

class SentimentAnalysisModel:
    def __init__(self, model_path='multilingual-sentiment', weights=None):
        if weights is not None:
            self.weights = weights.to(device)
        else:
            self.weights = torch.tensor([[-1, -0.5, 0, 0.5, 1]], device=device).unsqueeze(0).unsqueeze(-1)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, **inputs):
        with torch.no_grad():
            with torch.autocast(device):
                outputs = self.model(**inputs)
            predictions = outputs.logits
            probabilities = torch.softmax(predictions, dim=-1).to(device)
            sentiment_score = torch.matmul(probabilities, self.weights).squeeze()
        return sentiment_score.cpu()  # Move back to CPU for further processing

class CommentsDataset(Dataset):
    def __init__(self, comments):
        self.comments = comments

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.comments[idx]

def process_batch(batch, sentiment_model):
    tokenized_batch = sentiment_model.tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    tokenized_batch = {key: val.to(device) for key, val in tokenized_batch.items()}
    with torch.no_grad():
        scores = sentiment_model.forward(**tokenized_batch)
    del tokenized_batch
    torch.mps.empty_cache()
    return scores.tolist()

def process_all_batches(data_loader, sentiment_model):
    all_scores = []
    for batch in tqdm(data_loader, desc="Processing Batches", unit="batch"):
        batch_scores = process_batch(batch, sentiment_model)
        all_scores.extend(batch_scores)
    return all_scores
