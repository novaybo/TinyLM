import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import time
import argparse
import os
import math



class TinyLM(nn.Module):

    def __init__(self, vocab_size = 5000, embedding_dim = 256, hidden_dim = 512, num_layers = 3, dropout = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    

    def forward(self, x, hidden = None):

        batch_size, seq_length = x.size()

        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)

        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=-1
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1), lstm_out
        ).squeeze(1)

        context = context.unsqueeze(1).expand(-1, seq_length, -1)

        combined = lstm_out + context

        out = F.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.layer_norm(out)
        output = self.fc2(out)

        return output, hidden

    def init_hidden(self, batch_size):

        device = next(self.parameters()).device

        return (
            torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        )


def create_vocab(text):

    all_chars = list(text)
    char_counts = Counter(all_chars)

    sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
    chars = [ch for ch, _ in sorted_chars]

    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}

    return char2idx, idx2char, chars

def encode_text(text, char2idx, unknown_char = '�'):

    indices = []

    for ch in text:
        if ch in char2idx:
            indices.append(char2idx[ch])
        else:
            indices.append(char2idx.get(unknown_char, 0))
    return indices

def decode_text(indices, idx2char):

    return ''.join([idx2char[idx] for idx in indices])

def load_text_data(file_paths, max_chars = 100000):

    all_text = ""

    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            all_text += text + "\n\n"
        else:
            print(f"Warning: File {file_path} not found.")

    if len(all_text) > max_chars:
        print(f"Truncating from {len(all_text)} to {max_chars} characters")
        all_text = all_text[:max_chars]

    return all_text


class Char_Dataset(torch.utils.data.Dataset):

    def __init__(self, text, char2idx, min_length = 30, max_length = 100, stride = 20):

        self.text = text
        self.char2idx = char2idx
        self.min_length = min_length
        self.max_length = max_length
        self.stride = stride
        self.data = self._prepare_data()

    def _prepare_data(self):

        indices = encode_text(self.text, self.char2idx)
        sequences = []

        while i < len(indices) - self.min_length:
            seq_len = random.randint(self.min_length, self.max_length)
            seq_len = min(seq_len, len(indices) - i - 1)

            if seq_len < self.min_length:
                break

            seq = indices[i: + seq_len]
            target = indices[i + 1:i + seq_len + 1]

            sequences.append((seq, target))

            i += random.randint(self.stride // 2, self.stride)

        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    

def collate_func(batch):

    inputs, targets = zip(*batch)

    max_len = max(len(seq) for seq in inputs)

    padded_inputs = []
    padded_targets = []

    for input, target in zip(inputs, targets):
        pad_len = max_len - len(input)
        padded_inputs.append(F.pad(input, (0, pad_len), value = 0))
        padded_targets.append(F.pad(target, (0, pad_len), value = 0))

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def train_epoch(model, dataloader, criterion, optimizer, device,
                grad_clip = 1.0, print_every = 50):

    model.train()
    total_loss = 0
    total_chars = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, seq_length = inputs.shape

        hidden = model.init_hidden(batch_size)

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        mask = targets != 0
        outputs = outputs[mask]
        targets = targets[mask]

        if len(targets) > 0:
            loss = criterion(outputs, targets)
            loss.bacward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_chars += len(targets)

        if batch_idx % print_every == 0:
            avg_loss = total_loss / max(1, (batch_idx + 1))
            print(f"Batch {batch_idx}/{len(dataloader)},"
                  f"Loss: {loss.item() if 'loss' in locals else 0:.4f},"
                  f"Avg Loss: {avg_loss:.4f}")

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

def validate(model, dataloader, criterion, device):

    model.eval()
    total_loss = 0
    total_chars = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            hidden = model.init_hidden(batch_size)
            outputs, hidden = model(inputs, hidden)

            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            mask = targets != 0
            outputs = outputs[mask]
            targets = targets[mask]

            if len(targets) > 0:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total_chars += len(targets)

    return total_loss / max(1, total_chars) if total_chars > 0 else 0


def calculate_perplexity(loss):

    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


def generate_text(model, start_text, char2idx, idx2char,
                  device, length = 300, temprature = 0.7, top_k = 20):

    model.eval()

    generated = list(start_text)
    input_seq = torch.tensor(
        encode_text(start_text, char2idx)
    ).unsqueeze(0).to(device)

    hidden = None

    with torch.no_grad():

        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :] / temprature

            if top_k > 0:
                to_k_val = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=0)

            char_idx = torch.multinomial(probs, 1).item()
            generated.append(idx2char[char_idx])
            input_seq = torch.tensor([[char_idx]]).to(device)

            last_chars = ''.join(generated[-20:])
            if last_chars.count('(') > last_chars.count(')'):
                if idx2char[char_idx] == ')':
                    pass

    return ''.join(generated)


def main():

    parser = argparse.ArgumentParser(description = 'TinyLM training script')
    parser.add_argument('--data', nargs = '+', default = 'sample.txt', help = 'Path to training data')
    parser.add_argument('--epochs', type = int, default = 100, help = 'Number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size for training')
    parser.add_argument('--min_length', type = int, default = 30, help = 'Minimum sequence length')
    parser.add_argument('--max_length', type = int, default = 150, help = 'Maximum sequence length')
    parser.add_argument('--embedding_dim', type = int, default = 256, help = 'Dimension of character embeddings')
    parser.add_argument('--hidden_dim', type = int, default = 512, help = 'Dimension of LSTM hidden states')
    parser.add_argument('--num_layers', type = int, default = 3, help = 'Number of LSTM layers')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate for optimizer')
    parser.add_argument('--dropout', type = float, default = 0.2, help = 'Dropout rate')
    parser.add_argument('--save_model', type = str, default = 'tinylm.pth', help = 'Path to save the trained model')
    parser.add_argument('--max_chars', type = int, default = 100000, help = 'Maximum number of characters to read from data file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f"Loading data from {args.data}...")
    text = load_text_data(args.data, args.max_chars)
    print(f"Loaded {len(text)} characters.")
    print(f"Sample text: {text[:100]}...")

    char2idx, idx2char, chars = create_vocab(text)
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(chars[:50])}")

    char_counts = Counter(text)
    print(f"\nTop 20 characters by frequency:")
    for char, count in char_counts.most_common(20):
        if char == '\n':
            print(f"  '\\n': {count:,}")
        elif char == '\t':
            print(f"  '\\t': {count:,}")
        elif char == ' ':
            print(f"  ' ': {count:,}")
        else:
            print(f"  '{char}': {count:,}")

    print(f"\nCreating dataset...")
    dataset = Char_Dataset(
        text, char2idx,
        min_length = args.min_length,
        max_length = args.max_length,
        stride = args.max_length // 2
    )

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        collate_fn=collate_func,
        num_workers = 0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        collate_fn=collate_func,
        num_workers = 0
    )

    print(f"\nInitializing model...")
    model = TinyLM(
        vocab_size = vocab_size,
        embedding_dim = args.embedding_dim,
        hidden_dim = args.hidden_dim,
        num_layers = args.num_layers,
        dropout = args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = 'min', factor = 0.5, patience = 5, verbose = True
    )

    print(f"\nStarting training for {args.epochs} epochs...\n")
    train_losses = []
    val_losses = []
    perplexities = []
    best_val_loss = float('inf')

    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        perplexity = calculate_perplexity(val_loss)
        perplexities.append(perplexity)

        scheduler.step(val_loss)

        prompts = [
            "def ",
            "import ",
            "class ",
            "for i in ",
            "if __name__ == '__main__':",
            "if x > ",
            "print(",
            "# This function"
        ]

        sample = generate_text(
            model, random.choice(prompts), char2idx, idx2char, device,
            length = 100, temprature = 0.7, top_k = 20
        )

        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")
        print(f"Sample Text: {sample}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
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
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }, args.save_model)
            print(f"Model saved to {args.save_model}\n")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

        if (epoch + 1) % 10 == 0:
            print(f"\nProgress after {epoch + 1} epochs:")
            print(f" Best validation loss: {best_val_loss:.4f}")
            print(f" Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(train_losses, label='Train Loss', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss (log scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(perplexities, label='Perplexity', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Perplexity')
    axes[1, 0].set_title('Validation Perplexity')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot([i for i in range(len(train_losses))],
                    [optimizer.param_groups[0]['lr']] * len(train_losses),
                    color='red', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('coding_lm_training.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("FINAL TEXT GENERATION EXAMPLES")
    print("=" * 60)

    starters = [
        "def calculate_avarage(numbers):",
        "class User:",
        "for i in range(10):",
        "for item in data:",
        "if user_input == '':",
        "# This is a function to sort",
        "import pandas as pd",
        "def validate_email(email):"
        "async def fetch_data(url):",
    ]

    tempratures = [0.5, 0.7, 1.0]

    for starter in starters:
        print(f"\nStarter: '{starter}'")
        for temp in tempratures:
            generated_text = generate_text(
                model, starter, char2idx, idx2char, device,
                length = 200, temprature = temp, top_k = 25
            )
            print(f"Temp {temp}: {generated_text}\n")

    test_patterns = [
        ("Complete the function: def add(a, b):", 150),
        ("Write a loop: for i in range(", 100),
        ("Create a class: class Animal:", 200),
        ("Write error handling: try:", 150),
        ("Write a decorator: def logger(func):", 200),
    ]
    
    for prompt, length in test_patterns:
        print(f"\n{prompt}")
        generated = generate_text(
            model, prompt, char2idx, idx2char, device,
            length=length, temperature=0.7, top_k=20
        )
        print("-" * 50)
        print(generated)
        print("-" * 50)

    final_save_path = "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': char2idx,
        'idx2char': idx2char,
        'vocab_size': vocab_size,
        'config': {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        },
        'training_stats': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'perplexities': perplexities,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }
    }, final_save_path)

    print(f"\nFinal model saved to {final_save_path}")
    print("Training complete.")


if __name__ == '__main__':
    main()