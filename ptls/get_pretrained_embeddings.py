import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
import argparse
from pathlib import Path
import pyarrow.parquet as pq
import random
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig

def pad_tokens(batch, dtype):
    max_len = max(len(x) for x in batch)
    items = np.zeros((len(batch), max_len), dtype=dtype)
    for idx, item_ids in enumerate(batch):
        items[idx, :len(item_ids)] = item_ids
    return torch.from_numpy(items)


def collate_examples(batch):
    train_items, train_weights, test_items, test_weights = list(zip(*batch))
    return (
        pad_tokens(train_items, np.int64),
        pad_tokens(train_weights, np.float32),
        pad_tokens(test_items, np.int64),
        pad_tokens(test_weights, np.float32)
    )


class Embedder:
    def __init__(self, item_to_id, embeddings):
        assert len(item_to_id) == len(embeddings), 'Items and embeddings length mismatch'
        self.item_to_id = item_to_id
        self.embeddings = embeddings
        
class PairsDataset(Dataset):
    def __init__(self, items, weights, random_split, shuffle):
        self.items = [np.int64(item) for item in items]
        self.weights = [np.float32(weight) for weight in weights]
        self.random_split = random_split
        self.shuffle = shuffle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        items = self.items[idx]
        weights = self.weights[idx]

        if self.random_split:
            pivot = random.randint(1, len(items) - 1)
            indices = np.arange(len(items), dtype=np.int32)
            np.random.shuffle(indices)
            items = items[indices]
            weights = weights[indices]
        else:
            pivot = len(items)

        return items[:pivot], weights[:pivot], items[pivot:], weights[pivot:]

    def loader(self, batch_size):
        data_loader = DataLoader(
            self,
            collate_fn=collate_examples,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=self.shuffle
        )

        return data_loader

def metric_criterion(item, pair):
    margin = 0.4
    scale = 25
    scores = torch.mm(item, pair.transpose(0, 1))
    mask = torch.eye(scores.size()[0]).cuda()
    scores = scale * (scores - margin * mask)
    probs = (-scores.log_softmax(1) * mask).sum(dim=1)
    reverse_probs = (-scores.log_softmax(0) * mask).sum(dim=0)

    nll = (probs + reverse_probs) / 2
    return nll.mean()


class LightningModel(pl.LightningModule):
    def __init__(self, num_items, embedding_dim, train_loader, lr, num_updates):
        super().__init__()

        self.embeddings = nn.Embedding(num_items, embedding_dim)
        self.train_loader = train_loader
        self.num_updates = num_updates

        self.lr = lr

    def forward(self, input_ids, weights):
        weights = weights.unsqueeze(2)
        embedded = self.embeddings(input_ids) * weights
        embedded = embedded.sum(1) / weights.sum(1)

        return torch.nn.functional.normalize(embedded)

    def training_step(self, batch, batch_idx):
        left, weight, right, target_weight = batch

        left = self.forward(left, weight)
        right = self.forward(right, target_weight)
        loss = metric_criterion(left, right)
        self.log('loss', loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.train_loader

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0, self.num_updates)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'reduce_on_plateau': False,
            'frequency': 1
        }
        return [self.optimizer], [scheduler_config]


class MetricTrainer:
    def __init__(self, num_items, embedding_dim, num_epochs, batch_size):
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def fit(self, train_items, train_weights):
        grad_steps = 1
        lr = 1e-2

        train_loader = PairsDataset(train_items, train_weights, random_split=True, shuffle=True).loader(self.batch_size)

        effective_batch_size = self.batch_size * grad_steps
        num_updates = self.num_epochs * (len(train_items) + effective_batch_size - 1) // effective_batch_size

        self.model = LightningModel(
            self.num_items,
            self.embedding_dim,
            train_loader,
            lr,
            num_updates
        ).to('cuda')

        trainer = pl.Trainer(
            max_steps=num_updates,
            num_sanity_val_steps=0,
            accumulate_grad_batches=grad_steps,
            enable_checkpointing=False,
            gpus=[0],
        )
        trainer.fit(self.model)

    def predict(self, items, weights):
        loader = PairsDataset(items, weights, random_split=False, shuffle=False).loader(self.batch_size)
        result = []
        with torch.inference_mode():
            for batch in loader:
                result.append(self.model(batch[0], batch[1]).detach().cpu().numpy())

        return np.concatenate(result, axis=0)



@hydra.main(version_base='1.2', config_path=None)
def main(conf: DictConfig):
    if 'seed_everything' in conf:
        pl.seed_everything(conf.seed_everything)
    data_name = conf.data
    id_name = conf.id_name
    numeric_value = conf.numeric_value
    embeddings = conf.embeddings
    ARTIFACTS_PATH = Path('frozen_embeddings')
    for item_column, embedding_dim in embeddings.items():
        data = pq.read_table(
            data_name,
            columns=[id_name, item_column, numeric_value],
    )
        data_agg = data.select([id_name, item_column, numeric_value]).\
            group_by([id_name, item_column]).aggregate([(numeric_value, 'sum')])
        item_set = set(data_agg.select([item_column]).to_pandas()[item_column])
        item_dict = {url: idx for idx, url in enumerate(item_set)}
        user_set = set(data_agg.select([id_name]).to_pandas()[id_name])
        user_dict = {user: idx for idx, user in enumerate(user_set)}
        users = np.array(data_agg.select([id_name]).to_pandas()[id_name].map(user_dict))
        items = np.array(data_agg.select([item_column]).to_pandas()[item_column].map(item_dict))
        counts = np.array(data_agg.select([f'{numeric_value}_sum']).to_pandas()[f'{numeric_value}_sum'])
        df = pd.DataFrame.from_dict({id_name: users, 'items': items, 'counts': counts, 'ones': np.ones_like(counts)})
        orig_df = df.groupby(id_name).agg({'items': list, 'counts': list, 'ones': list}).reset_index()

        df = orig_df[orig_df['items'].apply(lambda x: len(x) > 1)]

        trainer = MetricTrainer(len(item_set), embedding_dim, 5, 2048)
        trainer.fit(df['items'].values, df['ones'].values)

        ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
        item_embedder = Embedder(item_dict, trainer.model.embeddings.weight.detach().cpu().numpy())
        with open(ARTIFACTS_PATH / f'{item_column}_{embedding_dim}.pickle', 'wb') as f:
            pickle.dump(item_embedder, f)
        

        
    
if __name__ == '__main__':
    main()