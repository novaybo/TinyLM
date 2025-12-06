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


PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'



class TinyLM(nn.Module):

    def __init__(self, vocab_size = 5000, embedding_dim = 128, hidden_dim = 256, num_layers = 2, dropout = 0.5, pad_idx = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
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

        out = F.relu(self.fc1(lstm_out))
        out = self.dropout_layer(out)
        out = self.layer_norm(out)
        output = self.fc2(out)

        return output, hidden

    def init_hidden(self, batch_size):

        device = next(self.parameters()).device

        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )


def create_vocab(text):

    all_chars = list(text)
    char_counts = Counter(all_chars)

    sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
    chars = [ch for ch, _ in sorted_chars]

    if PAD_TOKEN in chars:
        chars.remove(PAD_TOKEN)
    if UNK_TOKEN in chars:
        chars.remove(UNK_TOKEN)

    chars = [PAD_TOKEN, UNK_TOKEN] + chars

    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}

    return char2idx, idx2char, chars

def encode_text(text, char2idx, unknown_char = UNK_TOKEN):

    indices = []

    for ch in text:
        if ch in char2idx:
            indices.append(char2idx[ch])
        else:
            indices.append(char2idx.get(unknown_char, char2idx[PAD_TOKEN]))
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
        if len(indices) < self.min_length + 1:
            return sequences
        
        max_start = max(0, len(indices) - self.min_length - 1)
        num_samples = max(1, len(indices) // max(1, self.stride))

        for _ in range(num_samples):
            i = random.randint(0, max_start)
            seq_len = random.randint(self.min_length, min(self.max_length, len(indices) - i - 1))
            seq = indices[i:i + seq_len]
            target = indices[i + 1:i + seq_len + 1]

            if len(seq) >= self.min_length:
                sequences.append((seq, target))
        
        if len(sequences) < 1:
            i = 0
            while i < len(indices) - self.min_length:
                seq_len = random.randint(self.min_length, self.max_length)
                seq_len = min(seq_len, len(indices) - i - 1)
                if seq_len < self.min_length:
                    break
                seq = indices[i:i + seq_len]
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

    pad_idx = 0

    try:
        pad_idx = char2idx.get(PAD_TOKEN, 0)
    except NameError:
        pad_idx = 0

    for input, target in zip(inputs, targets):
        pad_len = max_len - len(input)
        padded_inputs.append(F.pad(input, (0, pad_len), value = pad_idx))
        padded_targets.append(F.pad(target, (0, pad_len), value = pad_idx))

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def train_epoch(model, dataloader, criterion, optimizer, device,
                grad_clip = 1.0, print_every = 50):

    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, seq_length = inputs.shape

        hidden = model.init_hidden(batch_size)

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % print_every == 0:
            avg_loss = total_loss / max(1, (batch_idx + 1))
            print(f"Batch {batch_idx}/{len(dataloader)},"
                  f"Loss: {loss.item() if 'loss' in locals() else 0:.4f},"
                  f"Avg Loss: {avg_loss:.4f}")

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

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

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0


def calculate_perplexity(loss):

    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


def generate_text(model, start_text, char2idx, idx2char,
                  device, length = 300, temperature = 0.8, top_k = 50, top_p = 0.8, rep_penalty = 1.0):

    model.eval()
    generated = list(start_text)
    
    model_device = next(model.parameters()).device
    device = device if device else model_device

    input_indices = encode_text(start_text, char2idx)
    input_seq = torch.tensor([input_indices], dtype=torch.long).to(device)

    hidden = None
    recent_tokens = []

    with torch.no_grad():

        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :] / max(1e-8, temperature)

            if rep_penalty != 1 and len(recent_tokens) > 0:
                for t in set(recent_tokens[-64:]):
                    logits[0, t] /= rep_penalty

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                topk_vals, topk_idx = torch.topk(logits, top_k_val, dim=-1)

                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(-1, topk_idx, False)
                logits[mask] = -float('Inf')

            if top_p and 0.0 < top_p < 1.0:
                probs_temp = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs_temp, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 0] = False
                mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_mask)
                logits[mask] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                probs = torch.ones_like(probs) / probs.size(-1)

            char_idx = torch.multinomial(probs.squeeze(0), 1).item()
            generated.append(idx2char[char_idx])
            recent_tokens.append(char_idx)
            
            input_seq = torch.tensor([[char_idx]]).to(device)

    return ''.join(generated)


def main():

    parser = argparse.ArgumentParser(description = 'TinyLM training script')
    parser.add_argument('--data', nargs = '+', default = ['/home/dante/VSCode/.virtual_env/projects/TinyLM/sample.txt'], help = 'Path to training data')
    parser.add_argument('--epochs', type = int, default = 200, help = 'Number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size for training')
    parser.add_argument('--min_length', type = int, default = 50, help = 'Minimum sequence length')
    parser.add_argument('--max_length', type = int, default = 200, help = 'Maximum sequence length')
    parser.add_argument('--embedding_dim', type = int, default = 256, help = 'Dimension of character embeddings')
    parser.add_argument('--hidden_dim', type = int, default = 512, help = 'Dimension of LSTM hidden states')
    parser.add_argument('--num_layers', type = int, default = 2, help = 'Number of LSTM layers')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate for optimizer')
    parser.add_argument('--dropout', type = float, default = 0.5, help = 'Dropout rate')
    parser.add_argument('--save_model', type = str, default = '/home/dante/VSCode/.virtual_env/projects/TinyLM/tinylm.pth', help = 'Path to save the trained model')
    parser.add_argument('--max_chars', type = int, default = 5000000, help = 'Maximum number of characters to read from data file')
    parser.add_argument('--temperature', type = float, default = 0.8, help = 'Temperature for text generation')
    parser.add_argument('--top_k', type = int, default = 50, help = 'Top-K sampling for text generation')
    parser.add_argument('--top_p', type = float, default = 0.8, help = 'Top-p (nucleus) sampling for text generation')
    parser.add_argument('--repeat_penalty', type = float, default = 4.0, help = 'Repetition penalty for text generation')
    parser.add_argument('--resume', type= str, default = None, help = 'Path to a checkpoint to resume training from')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    start_epoch = 0
    train_losses = []
    val_losses = []
    perplexities = []
    best_val_loss = float('inf')
    char2idx = None
    idx2char = None

    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        char2idx = checkpoint['char2idx']
        idx2char = checkpoint['idx2char']
        vocab_size = checkpoint['vocab_size']

        print(f"Loading data from {args.data}...")
        text = load_text_data(args.data, args.max_chars)
        print(f"Loaded {len(text)} characters from data.")

        model = TinyLM(
            vocab_size = vocab_size,
            embedding_dim = checkpoint.get('embedding_dim', args.embedding_dim),
            hidden_dim = checkpoint.get('hidden_dim', args.hidden_dim),
            num_layers = checkpoint.get('num_layers', args.num_layers),
            dropout = checkpoint.get('dropout', args.dropout),
            pad_idx=char2idx[PAD_TOKEN]
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])

        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        perplexities = checkpoint.get('perplexities', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"Resumed training from epoch {start_epoch}")
        print(f"Previous best validation loss: {best_val_loss:.4f}")

    else:
        print(f"Loading data from {args.data}...")
        text = load_text_data(args.data, args.max_chars)
        print(f"Loaded {len(text)} characters from data.")

        char2idx, idx2char, _ = create_vocab(text)
        vocab_size = len(char2idx)

        model = TinyLM(
            vocab_size = vocab_size,
            embedding_dim = args.embedding_dim,
            hidden_dim = args.hidden_dim,
            num_layers = args.num_layers,
            dropout = args.dropout,
            pad_idx=char2idx[PAD_TOKEN]
        ).to(device)

    print(f"Vocabulary size: {vocab_size}")

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

    pad_idx = char2idx[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 0.0001)

    if args.resume and os.path.exists(args.resume) and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state from checkpoint.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = 'min', factor = 0.5, patience = 5
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    print(f"\nStarting training for {args.epochs} epochs...\n")

    patience = 20
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
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
            "def   ",
            "import   ",
            "class   ",
            "for i in   ",
            "if __name__ == '__main__':   ",
            "if x >   ",
            "print(   ",
            "# This function   "
        ]

        sample = generate_text(
            model, random.choice(prompts), char2idx, idx2char, device,
            length = 100, temperature = args.temperature, top_k = args.top_k,
            top_p = args.top_p, rep_penalty = args.repeat_penalty
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
                'train_losses': train_losses,
                'val_losses': val_losses,
                'perplexities': perplexities,
                'best_val_loss': best_val_loss,
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
    plt.savefig('/home/dante/VSCode/.virtual_env/projects/TinyLM/coding_lm_training.png', dpi=300, bbox_inches='tight')
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
        "def validate_email(email):",
        "async def fetch_data(url):",
    ]

    temperatures = [0.5, 0.7, 1.0]

    for starter in starters:
        print(f"\nStarter: '{starter}'")
        for temp in temperatures:
            generated_text = generate_text(
                model, starter, char2idx, idx2char, device,
                length = 100, temperature = args.temperature, top_k = args.top_k,
                top_p = args.top_p, rep_penalty = args.repeat_penalty
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
            length=100, temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, rep_penalty=args.repeat_penalty
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