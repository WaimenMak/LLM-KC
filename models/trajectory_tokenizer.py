import torch
from typing import List, Tuple, Dict, Optional
import numpy as np

class TrajectoryTokenizer:
    """
    Custom tokenizer for trajectory data with T5-style tokens.
    """
    
    def __init__(self,
                 x_range: Tuple[int, int] = (1, 200),
                 y_range: Tuple[int, int] = (1, 200)):
        """
        Initialize tokenizer with feature ranges.
        
        Args:
            x_range: (min, max) for x coordinates
            y_range: (min, max) for y coordinates  
            t_range: (min, max) for time slots
            dow_range: (min, max) for day of week
            td_range: (min, max) for time delta bins
        """
        self.x_range = x_range
        self.y_range = y_range
        # Temporal features are now handled by separate embedding layers
        # in the transformer model, so they are NOT part of the vocabulary.
        
        # Build vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()
        
    def _build_vocab(self):
        """Build the complete vocabulary mapping."""
        vocab = []
        
        # Special tokens
        special_tokens = ['[PAD]', '[START]', '[END]', '[UNK]']
        vocab.extend(special_tokens)
        
        # Coordinate tokens
        for x in range(self.x_range[0], self.x_range[1] + 1):
            vocab.append(f'x_{x}')
        for y in range(self.y_range[0], self.y_range[1] + 1):
            vocab.append(f'y_{y}')
            
        # NOTE: t / dow / td are **NOT** tokenised any more – they will be
        # passed as separate integer features and looked-up via dedicated
        # embedding tables inside the model.
        
        # Create mappings
        for i, token in enumerate(vocab):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        self.vocab_size = len(vocab)
        
        # Define offsets for coordinate token calculation
        self.x_offset = self.token_to_id[f'x_{self.x_range[0]}'] - self.x_range[0]
        self.y_offset = self.token_to_id[f'y_{self.y_range[0]}'] - self.y_range[0]
        
        # Special token IDs
        self.pad_token_id = self.token_to_id['[PAD]']
        self.start_token_id = self.token_to_id['[START]']
        self.end_token_id = self.token_to_id['[END]']
        self.unk_token_id = self.token_to_id['[UNK]']
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Special tokens: PAD={self.pad_token_id}, START={self.start_token_id}, END={self.end_token_id}")

    def encode_coordinates(self, x: int, y: int) -> List[int]:
        """Encode a single (x, y) coordinate pair into tokens."""
        x_token = self.x_offset + x
        y_token = self.y_offset + y
        return [x_token, y_token]

    def encode_step(self, x: int, y: int) -> List[int]:
        """
        Encodes a single step, handling padding values.
        If coordinates are padding values (-999), returns padding tokens.
        """
        if x == -999 or y == -999:
            return [self.pad_token_id, self.pad_token_id]
        return self.encode_coordinates(x, y)
        
    def encode_trajectory(self, coords: List[Tuple[int, int]], add_special_tokens: bool = False) -> List[int]:
        """Encode a sequence of coordinates into a list of tokens."""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.start_token_id)
            
        for x, y in coords:
            tokens.extend(self.encode_step(x, y))
            
        if add_special_tokens:
            tokens.append(self.end_token_id)
            
        return tokens
    
    def encode_decoder_input(self, temporal_contexts: List[Tuple[int, int, int]], 
                           predicted_coords: List[Tuple[int, int]] = None) -> List[int]:
        """
        Encode the decoder's input sequence.
        
        Args:
            temporal_contexts: List of (t, dow, td) for each future step
            predicted_coords: List of (x, y) predictions so far (for autoregressive)
            
        Returns:
            List of token IDs for decoder input
        """
        tokens = [self.start_token_id]
        
        # First step: no coordinates yet → only START token present
        # Temporal context will be supplied to the model as separate indices
        # and embeddings, so we do NOT append it here.
        
        # Subsequent steps: add predicted coordinates + temporal context
        if predicted_coords:
            for i, (x, y) in enumerate(predicted_coords):
                tokens.extend(self.encode_coordinates(x, y))
        
        return tokens
    
    def decode_coordinates(self, token_ids: List[int]) -> List[Tuple[int, int]]:
        """Decode token IDs back to coordinate pairs."""
        coords = []
        
        # Remove special tokens
        filtered_ids = [tid for tid in token_ids if tid not in [self.pad_token_id, self.start_token_id, self.end_token_id]]
        
        # Extract coordinate pairs
        i = 0
        while i + 1 < len(filtered_ids):
            x_token = self.id_to_token.get(filtered_ids[i], '[UNK]')
            y_token = self.id_to_token.get(filtered_ids[i + 1], '[UNK]')
            
            if x_token.startswith('x_') and y_token.startswith('y_'):
                x = int(x_token.split('_')[1])
                y = int(y_token.split('_')[1])
                coords.append((x, y))
                i += 2
            else:
                i += 1
                
        return coords
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs to string tokens."""
        return [self.id_to_token.get(tid, '[UNK]') for tid in token_ids]