import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
import pickle
import os
import random
from typing import Dict, List, Tuple, Optional

class DynamicTrajectoryDataset(Dataset):
    """Dataset for training with dynamic trajectory splitting for data augmentation."""
    
    def __init__(self, h5_file_path: str, min_encoder_len: int = 20, min_decoder_len: int = 20,
                 max_encoder_len: Optional[int] = None, max_decoder_len: Optional[int] = None):
        """
        Args:
            h5_file_path: Path to HDF5 file containing full trajectories
            min_encoder_len: Minimum encoder sequence length
            min_decoder_len: Minimum decoder sequence length  
            max_encoder_len: Maximum encoder sequence length
            max_decoder_len: Maximum decoder sequence length
        """
        self.h5_file_path = h5_file_path
        self.min_encoder_len = min_encoder_len
        self.min_decoder_len = min_decoder_len
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        
        # Load metadata from HDF5 file
        with h5py.File(h5_file_path, 'r') as f:
            self.uids = f['uids'][:]
            self.cluster_indices = f['cluster_indices'][:]
            self.trajectory_lengths = f['trajectory_lengths'][:]
            self.num_trajectories = len(self.uids)
            
            # Check file type
            self.split_type = f.attrs.get('split_type', 'dynamic')
            if self.split_type != 'dynamic':
                print(f"Warning: Expected dynamic trajectory file, got {self.split_type}")
        
        # Filter trajectories that are long enough for splitting
        min_total_length = self.min_encoder_len + self.min_decoder_len
        self.valid_indices = [
            i for i, length in enumerate(self.trajectory_lengths) 
            if length >= min_total_length
        ]
        
        print(f"Loaded dynamic dataset: {len(self.valid_indices)}/{self.num_trajectories} valid trajectories")
        print(f"Min trajectory length required: {min_total_length}")
        print(f"Cluster range: {self.cluster_indices.min()} - {self.cluster_indices.max()}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the actual trajectory index
        traj_idx = self.valid_indices[idx]
        
        with h5py.File(self.h5_file_path, 'r') as f:
            # Load full trajectory
            trajectory = f['trajectories'][f'traj_{traj_idx}'][:]  # Shape: [seq_len, 5] (x,y,t,dow,td)
            uid = self.uids[traj_idx]
            cluster_idx = self.cluster_indices[traj_idx]
        
        # Dynamic splitting: randomly choose split point
        traj_len = len(trajectory)
        min_total = self.min_encoder_len + self.min_decoder_len
        
        # Ensure we have enough room for both encoder and decoder
        max_encoder_start = traj_len - min_total
        if max_encoder_start <= 0:
            # Fallback to minimum split
            encoder_len = self.min_encoder_len
            decoder_len = traj_len - encoder_len
        else:
            # Random encoder length between min and available space
            max_possible_encoder = min(
                traj_len - self.min_decoder_len,
                self.max_encoder_len if self.max_encoder_len else traj_len
            )
            encoder_len = random.randint(self.min_encoder_len, max_possible_encoder)
            
            # Decoder gets the rest (up to max_decoder_len)
            remaining_len = traj_len - encoder_len
            decoder_len = min(remaining_len, self.max_decoder_len if self.max_decoder_len else remaining_len)
        
        # Split trajectory
        encoder_part = trajectory[:encoder_len]  # [encoder_len, 5]
        decoder_part = trajectory[encoder_len:encoder_len + decoder_len]  # [decoder_len, 5]
        
        # Create encoder input: all 5 features [x, y, t, day_of_week, time_delta_encoded]
        encoder_input = encoder_part.astype(np.int16)
        
        # Create decoder input: all 5 features [x, y, t, day_of_week, time_delta_encoded] 
        # This is needed for teacher forcing during training
        decoder_input = decoder_part.astype(np.int16)  # Use all 5 columns, not just temporal
        
        # Create decoder target: coordinates only [x, y]
        decoder_target = decoder_part[:, :2].astype(np.int16)  # Take columns 0,1
        
        # Convert to tensors
        encoder_input = torch.tensor(encoder_input, dtype=torch.long)
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        decoder_target = torch.tensor(decoder_target, dtype=torch.long)
        
        # Create masks
        encoder_mask = torch.ones(len(encoder_input), dtype=torch.bool)
        decoder_mask = torch.ones(len(decoder_input), dtype=torch.bool)
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'decoder_target': decoder_target,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'uid': torch.tensor(uid, dtype=torch.long),
            'cluster_idx': torch.tensor(cluster_idx, dtype=torch.long),
            'split_type': 'dynamic'
        }

class FixedSplitDataset(Dataset):
    """Dataset for validation/testing with fixed encoder-decoder splits."""
    
    def __init__(self, h5_file_path: str, max_encoder_len: Optional[int] = None, 
                 max_decoder_len: Optional[int] = None):
        """
        Args:
            h5_file_path: Path to HDF5 file containing fixed splits
            max_encoder_len: Maximum encoder sequence length
            max_decoder_len: Maximum decoder sequence length
        """
        self.h5_file_path = h5_file_path
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        
        # Load metadata from HDF5 file
        with h5py.File(h5_file_path, 'r') as f:
            self.uids = f['uids'][:]
            self.cluster_indices = f['cluster_indices'][:]
            self.encoder_lengths = f['encoder_lengths'][:]
            self.decoder_lengths = f['decoder_lengths'][:]
            self.num_samples = len(self.uids)
            
            # Check file type
            self.split_type = f.attrs.get('split_type', 'fixed')
            if self.split_type != 'fixed':
                print(f"Warning: Expected fixed split file, got {self.split_type}")
        
        print(f"Loaded fixed split dataset: {self.num_samples} samples")
        print(f"Cluster range: {self.cluster_indices.min()} - {self.cluster_indices.max()}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as f:
            encoder_input = f['encoder_inputs'][f'sample_{idx}'][:]
            decoder_input = f['decoder_inputs'][f'sample_{idx}'][:]
            decoder_target = f['decoder_targets'][f'sample_{idx}'][:]
            
            uid = self.uids[idx]
            cluster_idx = self.cluster_indices[idx]
            
            # Apply length limits if specified
            if self.max_encoder_len and len(encoder_input) > self.max_encoder_len:
                encoder_input = encoder_input[-self.max_encoder_len:]  # Keep most recent
                
            if self.max_decoder_len and len(decoder_input) > self.max_decoder_len:
                decoder_input = decoder_input[:self.max_decoder_len]   # Keep earliest future
                decoder_target = decoder_target[:self.max_decoder_len]
            
            # Convert to tensors
            encoder_input = torch.tensor(encoder_input, dtype=torch.long)
            decoder_input = torch.tensor(decoder_input, dtype=torch.long)
            decoder_target = torch.tensor(decoder_target, dtype=torch.long)
            
            # Create masks
            encoder_mask = torch.ones(len(encoder_input), dtype=torch.bool)
            decoder_mask = torch.ones(len(decoder_input), dtype=torch.bool)
            
            return {
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
                'decoder_target': decoder_target,
                'encoder_mask': encoder_mask,
                'decoder_mask': decoder_mask,
                'uid': torch.tensor(uid, dtype=torch.long),
                'cluster_idx': torch.tensor(cluster_idx, dtype=torch.long),
                'split_type': 'fixed'
            }

def collate_mixed_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for both dynamic and fixed splits."""
    encoder_inputs = [item['encoder_input'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    decoder_targets = [item['decoder_target'] for item in batch]
    encoder_masks = [item['encoder_mask'] for item in batch]
    decoder_masks = [item['decoder_mask'] for item in batch]
    uids = [item['uid'] for item in batch]
    cluster_indices = [item['cluster_idx'] for item in batch]
    
    # Handle case where sequences are 1D (flattened) or 2D
    def pad_sequences_smart(sequences):
        if not sequences:
            return torch.empty(0)
        
        # Check if sequences are 1D or 2D
        if sequences[0].dim() == 1:
            # 1D sequences - use simple padding
            return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        else:
            # 2D sequences - pad sequence dimension
            max_len = max(seq.size(0) for seq in sequences)
            batch_size = len(sequences)
            feature_size = sequences[0].size(1)
            
            padded = torch.zeros((batch_size, max_len, feature_size), dtype=sequences[0].dtype)
            for i, seq in enumerate(sequences):
                seq_len = seq.size(0)
                padded[i, :seq_len] = seq
            
            return padded
    
    # Pad all sequences
    encoder_input_padded = pad_sequences_smart(encoder_inputs)
    decoder_input_padded = pad_sequences_smart(decoder_inputs)
    decoder_target_padded = pad_sequences_smart(decoder_targets)
    
    # Pad masks
    encoder_mask_padded = torch.nn.utils.rnn.pad_sequence(encoder_masks, batch_first=True, padding_value=False)
    decoder_mask_padded = torch.nn.utils.rnn.pad_sequence(decoder_masks, batch_first=True, padding_value=False)
    
    return {
        'encoder_input': encoder_input_padded,
        'decoder_input': decoder_input_padded,
        'decoder_target': decoder_target_padded,
        'encoder_mask': encoder_mask_padded,
        'decoder_mask': decoder_mask_padded,
        'uid': torch.stack(uids),
        'cluster_idx': torch.stack(cluster_indices),
        'batch_size': len(batch)
    }

class MixedTrajectoryDataManager:
    """Manager for loading mixed trajectory datasets (dynamic train + fixed val/test)."""
    
    def __init__(self, data_dir: str, city: str):
        """
        Args:
            data_dir: Directory containing processed trajectory HDF5 files and metadata
            city: City identifier
        """
        self.data_dir = data_dir
        self.city = city
        
        # Load metadata
        metadata_file = os.path.join(data_dir, f'{city}_mixed_trajectory_metadata.pkl')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
                self.cluster_info = self.metadata.get('cluster_info', {})
                print(f"✓ Loaded metadata with {self.cluster_info.get('num_clusters', 0)} clusters")
                print(f"✓ Training method: {self.metadata.get('train_method', 'unknown')}")
                print(f"✓ Val/Test method: {self.metadata.get('val_test_method', 'unknown')}")
        else:
            print(f"⚠️  No metadata file found at {metadata_file}")
            self.metadata = {}
            self.cluster_info = {}
    
    def get_data_loaders(self, batch_size: int = 32, 
                        min_encoder_len: int = 20, min_decoder_len: int = 20,
                        max_encoder_len: Optional[int] = None, max_decoder_len: Optional[int] = None,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get mixed data loaders: dynamic train + fixed val/test."""
        
        # File paths for mixed approach
        train_file = os.path.join(self.data_dir, f'train_{self.city}_full_trajectories.h5')
        val_file = os.path.join(self.data_dir, f'val_{self.city}_fixed_splits.h5')
        test_file = os.path.join(self.data_dir, f'test_{self.city}_fixed_splits.h5')
        
        loaders = {}
        
        # Training loader (dynamic splitting)
        if os.path.exists(train_file):
            print(f"📊 Creating dynamic training loader from {train_file}")
            train_dataset = DynamicTrajectoryDataset(
                train_file, 
                min_encoder_len=min_encoder_len,
                min_decoder_len=min_decoder_len,
                max_encoder_len=max_encoder_len, 
                max_decoder_len=max_decoder_len
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,  # Always shuffle training
                collate_fn=collate_mixed_batch,
                num_workers=num_workers,
                pin_memory=True
            )
            loaders['train'] = train_loader
            print(f"✓ Training loader: {len(train_dataset)} trajectories, batch size {batch_size}")
        else:
            print(f"❌ Training file not found: {train_file}")
            loaders['train'] = None
        
        # Validation loader (fixed splits)
        if os.path.exists(val_file):
            print(f"📊 Creating fixed validation loader from {val_file}")
            val_dataset = FixedSplitDataset(
                val_file,
                max_encoder_len=max_encoder_len,
                max_decoder_len=max_decoder_len
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,  # No shuffling for evaluation
                collate_fn=collate_mixed_batch,
                num_workers=num_workers,
                pin_memory=True
            )
            loaders['val'] = val_loader
            print(f"✓ Validation loader: {len(val_dataset)} samples, batch size {batch_size}")
        else:
            print(f"❌ Validation file not found: {val_file}")
            loaders['val'] = None
        
        # Test loader (fixed splits)
        if os.path.exists(test_file):
            print(f"📊 Creating fixed test loader from {test_file}")
            test_dataset = FixedSplitDataset(
                test_file,
                max_encoder_len=max_encoder_len,
                max_decoder_len=max_decoder_len
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,  # No shuffling for evaluation
                collate_fn=collate_mixed_batch,
                num_workers=num_workers,
                pin_memory=True
            )
            loaders['test'] = test_loader
            print(f"✓ Test loader: {len(test_dataset)} samples, batch size {batch_size}")
        else:
            print(f"❌ Test file not found: {test_file}")
            loaders['test'] = None
        
        return loaders['train'], loaders['val'], loaders['test']
    
    def get_cluster_info(self) -> Dict:
        """Get cluster information from metadata."""
        return self.cluster_info

# Convenience function
def load_mixed_trajectory_data(data_dir: str, city: str, batch_size: int = 32,
                              min_encoder_len: int = 20, min_decoder_len: int = 20,
                              max_encoder_len: Optional[int] = None, max_decoder_len: Optional[int] = None,
                              num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Quick function to load mixed trajectory data.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    manager = MixedTrajectoryDataManager(data_dir, city)
    return manager.get_data_loaders(
        batch_size=batch_size,
        min_encoder_len=min_encoder_len,
        min_decoder_len=min_decoder_len,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        num_workers=num_workers
    )

# Test the data loader
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    processed_dir = os.path.join(project_root, 'datasets', 'processed')
    
    print("🧪 Testing mixed trajectory data loader...")
    
    # Create data manager
    data_manager = MixedTrajectoryDataManager(processed_dir, city='D')
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_manager.get_data_loaders(
        batch_size=4,
        min_encoder_len=10,
        min_decoder_len=10,
        max_encoder_len=100,
        max_decoder_len=50,
        num_workers=0  # No multiprocessing for testing
    )
    
    # Test training batch (dynamic)
    if train_loader:
        print(f"\n🔄 Testing dynamic training batch...")
        batch = next(iter(train_loader))
        print(f"  Encoder input shape: {batch['encoder_input'].shape}")
        print(f"  Decoder input shape: {batch['decoder_input'].shape}")
        print(f"  Decoder target shape: {batch['decoder_target'].shape}")
        print(f"  UIDs: {batch['uid']}")
        print(f"  Clusters: {batch['cluster_idx']}")
    
    # Test validation batch (fixed)
    if val_loader:
        print(f"\n🔄 Testing fixed validation batch...")
        batch = next(iter(val_loader))
        print(f"  Encoder input shape: {batch['encoder_input'].shape}")
        print(f"  Decoder input shape: {batch['decoder_input'].shape}")
        print(f"  Decoder target shape: {batch['decoder_target'].shape}")
    
    print("✅ Data loader test completed!") 