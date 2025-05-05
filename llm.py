import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    def __init__(self, d_model: int, max_seq_length: int = 2048):
        super().__init__()
        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor."""
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (heads, depth)."""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
    
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge the heads back."""
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, d_k)
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate attention weights and values."""
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Apply attention weights to values
        output = torch.matmul(weights, v)
        return output
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        batch_size = q.size(0)
        
        # Linear projections
        q = self.w_q(q)  # (batch_size, seq_len, d_model)
        k = self.w_k(k)  # (batch_size, seq_len, d_model)
        v = self.w_v(v)  # (batch_size, seq_len, d_model)
        
        # Split heads
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, d_k)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, d_k)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Apply attention
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len_q, d_k)
        
        # Merge heads
        output = self.merge_heads(attn_output)  # (batch_size, seq_len_q, d_model)
        
        # Final linear layer
        output = self.w_o(output)  # (batch_size, seq_len_q, d_model)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward layer in transformer block."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SimpleLanguageModel(nn.Module):
    """A simplified transformer-based language model."""
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 768, 
                 num_layers: int = 12, 
                 num_heads: int = 12, 
                 d_ff: int = 3072, 
                 max_seq_length: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embedding weights with LM head weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        # Get embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create attention mask for self-attention if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create causal mask (for auto-regressive generation)
        seq_length = input_ids.size(1)
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), device=input_ids.device), diagonal=1).bool()
        causal_mask = ~causal_mask  # Invert to get the correct mask
        
        # Combine with attention mask
        if attention_mask is not None:
            # Extend attention_mask to match the shape required for attention
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            # Combine with causal mask
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            causal_mask = causal_mask.to(dtype=torch.float32)
            causal_mask = (1.0 - causal_mask) * -10000.0
            
            combined_mask = torch.max(extended_attention_mask, causal_mask)
        else:
            combined_mask = None
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, combined_mask)
        
        # Apply final layer norm
        x = self.final_layer_norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits

class TextDataset(Dataset):
    """Simple dataset for text sequences."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, 
                                 padding="max_length", return_tensors="pt")
            self.inputs.append({
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "labels": encodings["input_ids"].squeeze()
            })
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.inputs[idx]

def train_model(model: nn.Module, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
               num_epochs: int = 3, device: str = "cuda"):
    """Train the language model."""
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), 
                          shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def generate_text(model: nn.Module, tokenizer, prompt: str, max_length: int = 50, 
                 temperature: float = 1.0, top_k: int = 50, device: str = "cuda"):
    """Generate text from the trained model."""
    model.to(device)
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, top_k)[0][..., -1, None].repeat(1, next_token_logits.size(-1))
                next_token_logits = torch.where(next_token_logits < indices_to_remove, 
                                              torch.ones_like(next_token_logits) * -float('Inf'), 
                                              next_token_logits)
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we predict EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Example usage (would need a tokenizer and data to actually run)
"""
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create a small model
model = SimpleLanguageModel(
    vocab_size=tokenizer.vocab_size,
    d_model=256,  # Much smaller than production models
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    max_seq_length=512
)

# Sample data (in a real scenario, you'd have a large dataset)
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    # ... more training texts
]

# Create dataset and dataloader
dataset = TextDataset(texts, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Train model
model = train_model(model, dataloader, optimizer, num_epochs=3)

# Generate text
generated_text = generate_text(model, tokenizer, "The quick brown")
print(generated_text)
"""