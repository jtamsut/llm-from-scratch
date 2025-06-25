import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# Constants
FILE = "/Users/jon/Desktop/llm-from-scratch/data/file1.txt"

class DatasetCreator(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must greater than max_length"

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def init_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = DatasetCreator(txt, tokenizer, max_length, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

if __name__ == "__main__":
    vocab_size = 50257
    output_dim = 256
    context_length = 1024
    batch_size = 8
    max_length = 4

    with open(FILE, "r", encoding="utf-8") as file:
        text = file.read()

    dataloader = init_dataloader(
        text,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
    )

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    for batch in dataloader:
        x, y = batch 

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        print(input_embeddings)

        break
