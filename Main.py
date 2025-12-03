import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import time
import argparse



class TinyLM(nn.Module):

    def __init__(self, vocab_size = 5000, embedding_dim = 256, hidden_dim = 512, num_layers = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    

    def froward(self, x, hidden = None):

        batch_size = x.size(0)

        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size):

        device = next(self.parameters()).device

        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))


def create_vocab(text):

    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}

    return char2idx, idx2char, chars

def encode_text(text, char2idx):

    return [char2idx[ch] for ch in text]

def decode_text(encoded_text, idx2char):

    return ''.join([idx2char[idx] for idx in encoded_text])

def load_text_data(file_path, max_chars = 20000):

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()[:max_chars]

    return text


class Char_Dataset(torch.utils.data.Dataset):

    def __init__(self, text, char2idx, seq_length = 50, stride = 25):

        self.text = text
        self.char2idx = char2idx
        self.seq_length = seq_length
        self.stride = stride
        self.data = self._prepare_data()

    def _prepare_data(self):

        indices = encode_text(self.text, self.char2idx)
        sequences = []

        for i in range(0, len(indices) - self.seq_length, self.stride):
            seq = indices[i:i + self.seq_length]
            target = indices[i + 1:i + self.seq_length + 1]
            sequences.append((seq, target))

        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        hidden = model.init_hidden(batch_size)

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f" Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = model.init_hidden(batch_size)
            outputs, hidden = model(inputs, hidden)

            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def generate_text(model, start_text, char2idx, idx2char,
                  device, length = 200, temprature = 0.8, top_k = 10):

    model.eval()

    generated = list(start_text)
    input_seq = torch.tensor(
        [char2idx[ch] for ch in start_text]
    ).unsqueeze(0).to(device)

    hidden = None

    with torch.no_grad():

        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :] / temprature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=0)

            char_idx = torch.multinomial(probs, 1).item()
            generated.append(idx2char[char_idx])
            input_seq = torch.tensor([[char_idx]]).to(device)

    return ''.join(generated)


def main():

    parser = argparse.ArgumentParser(description = 'TinyLM training script')
    parser.add_argument('--data', type = str, default = 'sample.txt', help = 'Path to training data')
    parser.add_argument('--epochs', type = int, default = 30, help = 'Number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size for training')
    parser.add_argument('--seq_length', type = int, default = 50, help = 'Sequence length for training')
    parser.add_argument('--embedding_dim', type = int, default = 256, help = 'Dimension of character embeddings')
    parser.add_argument('--hidden_dim', type = int, default = 512, help = 'Dimension of LSTM hidden states')
    parser.add_argument('--num_layers', type = int, default = 3, help = 'Number of LSTM layers')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate for optimizer')
    parser.add_argument('--save_model', type = str, default = 'tinylm.pth', help = 'Path to save the trained model')
    parser.add_argument('--max_chars', type = int, default = 20000, help = 'Maximum number of characters to read from data file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f"Loading data from {args.data}...")
    text = load_text_data(args.data, max_chars = args.max_chars)
    print(f"Loaded {len(text)} characters.")
    print(f"Sample text: {text[:100]}...")

    char2idx, idx2char, chars = create_vocab(text)
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(chars)}")

    dataset = Char_Dataset(text, char2idx, seq_length = args.seq_length)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = args.batch_size
    )

    print(f"\nInitializing model...")
    model = TinyLM(
        vocab_size = vocab_size,
        embedding_dim = args.embedding_dim,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = 'min', factor = 0.5, patience = 3, verbose = True
    )

    print(f"\nStarting training for {args.epochs} epochs...\n")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        sample_text = generate_text(
            model, "The", char2idx, idx2char, device,
            length = 100, temprature = 0.7
        )

        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        print(f"Sample Text: {sample_text}\n")

        if val_loss < best_val_loss:
            best_vall_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'char2idx': char2idx,
                'idx2char': idx2char,
                'vocab_size': vocab_size,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers
            }, args.save_model)
            print(f"Model saved to {args.save_model}\n")

    plt.figure(figsize = (12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label = 'Train Loss')
    plt.plot(val_losses, label = 'Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label = 'Train Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (log scale)')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print("\n" + "=" * 60)
    print("FINAL TEXT GENERATION EXAMPLES")
    print("=" * 60)

    starters = ["The", "i think", "in the", "hello"]
    tempratures = [0.5, 0.7, 1.0]

    for starter in starters:
        print(f"\nStarter: '{starter}'")
        for temp in tempratures:
            generated_text = generate_text(
                model, starter, char2idx, idx2char, device,
                length = 200, temprature = temp
            )
            print(f"Temp {temp}: {generated_text}\n")


if __name__ == '__main__':
    main()