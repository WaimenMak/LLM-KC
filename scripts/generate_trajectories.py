import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional
import time
import pickle
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trajectory_tokenizer import TrajectoryTokenizer
from models.trajectory_transformer import TrajectoryT5
# from models.trajectory_transformer_sep import TrajectoryT5Sep as TrajectoryT5

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, num_clusters: int = 100) -> TrajectoryT5:
    """Load model from checkpoint with cluster support."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get training args
    training_args = checkpoint.get('args', {})
    model_name = training_args.get('model_name', 't5-small')
    
    # Initialize tokenizer and model with cluster support
    tokenizer = TrajectoryTokenizer()
    model = TrajectoryT5(
        tokenizer, 
        model_name=model_name, 
        from_pretrained=False,
        num_clusters=num_clusters
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully (Epoch: {checkpoint.get('epoch', 'Unknown')})")
    print(f"✓ Model supports {num_clusters} clusters")
    return model

def load_enriched_data(parquet_path: str, target_days: Tuple[int, int], 
                      target_uids: Optional[Tuple[int, int]] = None,
                      max_encoder_steps: int = 250) -> List[dict]:
    """
    Load enriched Parquet data and prepare for generation.
    
    Args:
        parquet_path: Path to enriched Parquet file
        target_days: (start_day, end_day) for trajectory generation
        target_uids: Optional (start_uid, end_uid) to filter users
        max_encoder_steps: Maximum number of encoder trajectory steps
        
    Returns:
        List of generation data prepared for each user
    """
    print(f"📁 Loading enriched data from {parquet_path}")
    
    # Load the entire enriched dataset (fast with Parquet)
    df = pd.read_parquet(parquet_path)
    
    print(f"✓ Loaded {len(df):,} trajectory points for {df['uid'].nunique():,} users")
    print(f"📅 Day range: {df['d'].min()} - {df['d'].max()}")
    print(f"🏷️  Clusters: {df['cluster_idx'].nunique():,} unique clusters")
    
    # Filter by user ID range if specified
    if target_uids:
        start_uid, end_uid = target_uids
        df = df[(df['uid'] >= start_uid) & (df['uid'] <= end_uid)]
        print(f"🔍 Filtered to UIDs {start_uid}-{end_uid}: {df['uid'].nunique():,} users")
    
    # Separate target period (for generation) and historical data
    start_day, end_day = target_days
    target_df = df[(df['d'] >= start_day) & (df['d'] <= end_day)].copy()
    historical_df = df[df['d'] < start_day].copy()
    
    print(f"🎯 Target period: days {start_day}-{end_day}, {len(target_df):,} points")
    print(f"📚 Historical data: days < {start_day}, {len(historical_df):,} points")
    
    if len(target_df) == 0:
        raise ValueError(f"No data found in target period {start_day}-{end_day}")
    
    # Prepare generation data for each user
    generation_data = []
    target_users = target_df['uid'].unique()
    
    print(f"🔄 Preparing generation data for {len(target_users):,} users...")
    
    for uid in tqdm(target_users, desc="Processing users"):
        # Get user's target trajectory (what we want to generate)
        user_target = target_df[target_df['uid'] == uid].sort_values(['d', 't'])
        
        # Get user's historical trajectory (encoder input)
        user_history = historical_df[historical_df['uid'] == uid].sort_values(['d', 't'])
        
        if len(user_history) == 0:
            print(f"⚠️  Warning: No historical data for user {uid}, skipping")
            continue
        
        # Limit historical data to max_encoder_steps
        if len(user_history) > max_encoder_steps:
            user_history = user_history.tail(max_encoder_steps)
        
        # Extract features
        cluster_idx = user_target['cluster_idx'].iloc[0]  # Assume cluster is consistent per user
        
        # Historical trajectory (encoder input): [x, y, t, dow, td]
        past_coords = [(int(row['x']), int(row['y'])) for _, row in user_history.iterrows()]
        past_temporal = [(int(row['t']), int(row['day_of_week']), int(row['time_delta_encoded'])) 
                        for _, row in user_history.iterrows()]
        
        # Target trajectory temporal context (decoder input): [t, dow, td]
        future_temporal = [(int(row['t']), int(row['day_of_week']), int(row['time_delta_encoded'])) 
                          for _, row in user_target.iterrows()]
        
        # Ground truth coordinates (for comparison)
        target_coords = [(int(row['x']), int(row['y'])) for _, row in user_target.iterrows()]
        
        # Original day/time information for output
        target_days_times = [(int(row['d']), int(row['t'])) for _, row in user_target.iterrows()]
        
        generation_data.append({
            'uid': int(uid),
            'cluster_idx': int(cluster_idx),
            'past_coords': past_coords,
            'past_temporal': past_temporal,
            'future_temporal': future_temporal,
            'target_coords': target_coords,
            'target_days_times': target_days_times,
            'encoder_length': len(past_coords),
            'decoder_length': len(future_temporal)
        })
    
    print(f"✅ Successfully prepared data for {len(generation_data):,} users")
    
    # Show some statistics
    encoder_lengths = [data['encoder_length'] for data in generation_data]
    decoder_lengths = [data['decoder_length'] for data in generation_data]
    
    print(f"📊 Data statistics:")
    print(f"   Encoder lengths: avg={np.mean(encoder_lengths):.1f}, min={min(encoder_lengths)}, max={max(encoder_lengths)}")
    print(f"   Decoder lengths: avg={np.mean(decoder_lengths):.1f}, min={min(decoder_lengths)}, max={max(decoder_lengths)}")
    
    return generation_data

def generate_trajectories_from_data(model: TrajectoryT5, generation_data: List[dict], 
                                  device: torch.device, generation_strategy: str = 'greedy',
                                  top_k: int = 10, beam_width: int = 5, 
                                  temperature: float = 1.0,
                                  snap_mode: str = 'none',
                                  reachable_points: Optional[np.ndarray] = None) -> List[dict]:
    """
    Generate trajectories for prepared data using optimized generation methods.
    ...
    """
    model.eval()
    
    print(f"\n🔮 Generating trajectories using {generation_strategy} strategy (OPTIMIZED)")
    if snap_mode != 'none':
        print(f"📌 Snap mode: {snap_mode}")
    print(f"👥 Processing {len(generation_data):,} users")
    print("="*70)
    
    def snap_to_nearest(points: List[Tuple[int,int]], candidates_np: np.ndarray) -> List[Tuple[int,int]]:
        if candidates_np is None or len(points) == 0 or candidates_np.size == 0:
            return points
        pts = np.array(points, dtype=np.int32)
        snapped = []
        for p in pts:
            # squared distances for speed
            d2 = (candidates_np[:,0] - p[0])**2 + (candidates_np[:,1] - p[1])**2
            j = int(np.argmin(d2))
            snapped.append((int(candidates_np[j,0]), int(candidates_np[j,1])))
        return snapped

    results = []
    total_generation_time = 0.0
    successful_generations = 0
    
    with torch.inference_mode():
        for i, user_data in enumerate(tqdm(generation_data, desc="Generating trajectories")):
            try:
                uid = user_data['uid']
                cluster_idx = user_data['cluster_idx']
                past_coords = user_data['past_coords']
                past_temporal = user_data['past_temporal']
                future_temporal = user_data['future_temporal']
                target_coords = user_data['target_coords']
                target_days_times = user_data['target_days_times']
                
                if len(past_coords) == 0 or len(future_temporal) == 0:
                    print(f"⚠️  Warning: No valid data for UID {uid}")
                    continue
                
                start_time = time.time()
                
                if generation_strategy == 'sampling':
                    predicted_trajectories = model.generate_trajectory_with_sampling(
                        batch_past_coords=[past_coords],
                        batch_past_temporal=[past_temporal],
                        batch_future_temporal=[future_temporal],
                        batch_cluster_indices=[cluster_idx],
                        device=device,
                        top_k=top_k,
                        temperature=temperature
                    )
                elif generation_strategy == 'beam':
                    predicted_trajectories = model.generate_trajectory_with_beam_search(
                        batch_past_coords=[past_coords],
                        batch_past_temporal=[past_temporal],
                        batch_future_temporal=[future_temporal],
                        batch_cluster_indices=[cluster_idx],
                        device=device,
                        beam_width=beam_width
                    )
                else:  # greedy
                    predicted_trajectories = model.generate_trajectory(
                        batch_past_coords=[past_coords],
                        batch_past_temporal=[past_temporal],
                        batch_future_temporal=[future_temporal],
                        batch_cluster_indices=[cluster_idx],
                        device=device
                    )
                
                generation_time = time.time() - start_time
                total_generation_time += generation_time
                
                pred_coords = predicted_trajectories[0]

                # Snap post-processing
                if snap_mode == 'history':
                    if len(past_coords) > 0:
                        hist_np = np.unique(np.array(past_coords, dtype=np.int32), axis=0)
                        pred_coords = snap_to_nearest(pred_coords, hist_np)
                elif snap_mode == 'reachable':
                    pred_coords = snap_to_nearest(pred_coords, reachable_points)

                result = {
                    'uid': uid,
                    'cluster_idx': cluster_idx,
                    'predicted_coords': pred_coords,
                    'target_coords': target_coords,
                    'target_days_times': target_days_times,
                    'generation_time': generation_time,
                    'encoder_length': user_data['encoder_length'],
                    'decoder_length': user_data['decoder_length']
                }
                results.append(result)
                successful_generations += 1
                
                if (i + 1) % 50 == 0:
                    avg_time = total_generation_time / (i + 1)
                    print(f"🔄 Processed {i + 1:,}/{len(generation_data):,} users, "
                          f"successful: {successful_generations:,}, "
                          f"avg time: {avg_time:.3f}s/user")
                
            except Exception as e:
                print(f"❌ Error processing user {user_data['uid']}: {e}")
                continue
    
    print(f"\n✅ GENERATION COMPLETE")
    print(f"👥 Total users processed: {len(generation_data):,}")
    print(f"🎯 Successful generations: {successful_generations:,}")
    print(f"📈 Success rate: {successful_generations/len(generation_data)*100:.1f}%")
    print(f"⏱️  Total generation time: {total_generation_time:.2f}s")
    print(f"🚀 Average time per user: {total_generation_time/len(generation_data):.3f}s")
    
    return results

def save_results_to_csv(results: List[dict], output_file: str):
    """Save generation results to CSV in original data format."""
    print(f"\n💾 Saving results to {output_file}")
    
    # Convert results to original data format
    output_data = []
    
    for result in results:
        uid = result['uid']
        cluster_idx = result['cluster_idx']
        pred_coords = result['predicted_coords']
        target_days_times = result['target_days_times']
        
        # Create entries for each predicted coordinate
        for (pred_x, pred_y), (d, t) in zip(pred_coords, target_days_times):
            output_data.append({
                'uid': uid,
                'd': d,
                't': t,
                'x': pred_x,
                'y': pred_y,
                'cluster_idx': cluster_idx
            })
    
    if output_data:
        df = pd.DataFrame(output_data)
        df = df.sort_values(['uid', 'd', 't']).reset_index(drop=True)
        df.to_csv(output_file, index=False)
        
        print(f"✓ Saved {len(output_data):,} trajectory points")
        print(f"📊 Data summary:")
        print(f"   Unique UIDs: {df['uid'].nunique():,}")
        print(f"   Unique clusters: {df['cluster_idx'].nunique():,}")
        print(f"   Day range: {df['d'].min()} - {df['d'].max()}")
        print(f"   Time range: {df['t'].min()} - {df['t'].max()}")
        print(f"   X coordinate range: {df['x'].min()} - {df['x'].max()}")
        print(f"   Y coordinate range: {df['y'].min()} - {df['y'].max()}")
        
        # Show sample
        print(f"\n📝 Sample of generated data:")
        print(df.head(10))
        
    else:
        print("❌ No data to save")

def save_comparison_results(results: List[dict], output_file: str):
    """Save detailed comparison between predictions and ground truth."""
    comparison_file = output_file.replace('.csv', '_comparison.csv')
    print(f"\n📊 Saving comparison results to {comparison_file}")
    
    comparison_data = []
    
    for result in results:
        uid = result['uid']
        cluster_idx = result['cluster_idx']
        pred_coords = result['predicted_coords']
        target_coords = result['target_coords']
        target_days_times = result['target_days_times']
        
        # Calculate metrics for each point
        for i, ((pred_x, pred_y), (true_x, true_y), (d, t)) in enumerate(
            zip(pred_coords, target_coords, target_days_times)):
            
            # Calculate Euclidean distance error
            distance_error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            
            comparison_data.append({
                'uid': uid,
                'cluster_idx': cluster_idx,
                'd': d,
                't': t,
                'step': i,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'true_x': true_x,
                'true_y': true_y,
                'distance_error': distance_error,
                'encoder_length': result['encoder_length'],
                'decoder_length': result['decoder_length'],
                'generation_time': result['generation_time']
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.to_csv(comparison_file, index=False)
        
        # Calculate summary statistics
        mean_error = df['distance_error'].mean()
        median_error = df['distance_error'].median()
        
        print(f"✓ Saved detailed comparison for {len(comparison_data):,} points")
        print(f"📈 Error statistics:")
        print(f"   Mean distance error: {mean_error:.2f}")
        print(f"   Median distance error: {median_error:.2f}")
        print(f"   90th percentile error: {df['distance_error'].quantile(0.9):.2f}")
        print(f"   99th percentile error: {df['distance_error'].quantile(0.99):.2f}")

def visualize_sample_attention(model, sample_user_data, max_enc_len, max_dec_len, device):
    """Visualize attention patterns for a specific sample from user data."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    uid = sample_user_data['uid']
    cluster_idx = sample_user_data['cluster_idx']
    past_coords = sample_user_data['past_coords']
    past_temporal = sample_user_data['past_temporal']
    future_temporal = sample_user_data['future_temporal']
    target_coords = sample_user_data['target_coords']
    
    print(f"🎨 Visualizing attention for UID {uid}...")
    
    try:
        # Create a batch-like structure from sample_user_data
        # Convert to tensors and add batch dimension
        past_coords_tensor = torch.tensor(past_coords, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 2]
        past_temporal_tensor = torch.tensor(past_temporal, dtype=torch.long).unsqueeze(0)  # [1, seq_len, 3]
        future_temporal_tensor = torch.tensor(future_temporal, dtype=torch.long).unsqueeze(0)  # [1, seq_len, 3]
        target_coords_tensor = torch.tensor(target_coords, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 2]
        cluster_tensor = torch.tensor([cluster_idx], dtype=torch.long)  # [1]
        
        # Create encoder input: combine coordinates and temporal features
        encoder_seq_len = past_coords_tensor.shape[1]
        encoder_input = torch.cat([past_coords_tensor, past_temporal_tensor], dim=-1)  # [1, seq_len, 5]
        encoder_mask = torch.ones(1, encoder_seq_len, dtype=torch.bool)
        
        # Create decoder input: temporal features only (for context)
        decoder_seq_len = future_temporal_tensor.shape[1]
        decoder_input = future_temporal_tensor  # [1, seq_len, 3]
        decoder_mask = torch.ones(1, decoder_seq_len, dtype=torch.bool)
        
        # Create decoder target: coordinates only
        decoder_target = target_coords_tensor  # [1, seq_len, 2]
        
        # Create a batch-like dictionary with all required keys
        batch = {
            'uid': cluster_tensor,
            'cluster_idx': cluster_tensor,
            'encoder_input': encoder_input,
            'encoder_mask': encoder_mask,
            'decoder_input': decoder_input,
            'decoder_mask': decoder_mask,
            'decoder_target': decoder_target,
            'batch_size': 1  # This key was missing!
        }
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Prepare tokenized batch using the model's method
        tokenized_batch = model.prepare_batch_data(batch)
        
        # Store and enable attention output
        orig_att_setting = model.model.config.output_attentions
        model.model.config.output_attentions = True
        
        try:
            with torch.no_grad():
                # Extract components for forward pass (following quick_evaluate.py approach)
                encoder_input_ids = tokenized_batch['encoder_input_ids']
                encoder_temporal = tokenized_batch['encoder_temporal']
                encoder_attention_mask = tokenized_batch['encoder_attention_mask']
                decoder_input_ids = tokenized_batch['decoder_input_ids']
                decoder_temporal = tokenized_batch['decoder_temporal']
                decoder_attention_mask = tokenized_batch['decoder_attention_mask']
                encoder_clusters = tokenized_batch['encoder_clusters']
                decoder_clusters = tokenized_batch['decoder_clusters']
                labels = tokenized_batch.get('labels', None)
                
                # Get embeddings manually (similar to model.forward)
                encoder_coord_embeds = model.model.encoder.embed_tokens(encoder_input_ids)
                decoder_coord_embeds = model.model.decoder.embed_tokens(decoder_input_ids)
                
                # Temporal embeddings  
                encoder_t_embeds = model.t_embedding(encoder_temporal[:, :, 0])
                encoder_dow_embeds = model.dow_embedding(encoder_temporal[:, :, 1])
                encoder_td_embeds = model.td_embedding(encoder_temporal[:, :, 2])
                
                decoder_t_embeds = model.t_embedding(decoder_temporal[:, :, 0])
                decoder_dow_embeds = model.dow_embedding(decoder_temporal[:, :, 1])
                decoder_td_embeds = model.td_embedding(decoder_temporal[:, :, 2])
                
                # Cluster embeddings
                encoder_cluster_embeds = model.cluster_embedding(encoder_clusters)
                decoder_cluster_embeds = model.cluster_embedding(decoder_clusters)
                
                # Combine embeddings
                encoder_embeds = (encoder_coord_embeds + encoder_t_embeds + encoder_dow_embeds + 
                                encoder_td_embeds + encoder_cluster_embeds)
                decoder_embeds = (decoder_coord_embeds + decoder_t_embeds + decoder_dow_embeds + 
                                decoder_td_embeds + decoder_cluster_embeds)
                
                # Call the underlying T5 model directly with output_attentions=True
                outputs = model.model(
                    inputs_embeds=encoder_embeds,
                    attention_mask=encoder_attention_mask,
                    decoder_inputs_embeds=decoder_embeds,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                    output_attentions=True
                )
                
                # Extract cross-attention (decoder attending to encoder)
                cross_attentions = outputs.cross_attentions
                
                # Get valid lengths
                enc_mask = (encoder_attention_mask[0] == 1).cpu().numpy()
                dec_mask = (decoder_attention_mask[0] == 1).cpu().numpy()
                valid_enc_len = enc_mask.sum()
                valid_dec_len = dec_mask.sum()
                
                print(f"   📏 Total tokens - Encoder: {valid_enc_len}, Decoder: {valid_dec_len}")
                
                if cross_attentions and len(cross_attentions) > 0:
                    # Get full attention matrix from last layer, first head
                    full_attention = cross_attentions[-1][0, 0].cpu().numpy()
                    full_attention = full_attention[:valid_dec_len, :valid_enc_len]
                    
                    # Decode all tokens
                    enc_tokens = model.tokenizer.decode_tokens(encoder_input_ids[0][:valid_enc_len].cpu().tolist())
                    dec_tokens = model.tokenizer.decode_tokens(decoder_input_ids[0][:valid_dec_len].cpu().tolist())
                    
                    # Show subset for readability (last 30 encoder tokens, first 30 decoder tokens)
                    max_enc_show = min(max_enc_len, valid_enc_len)
                    max_dec_show = min(max_dec_len, valid_dec_len)
                    
                    # Select encoder subset (last N tokens - most recent trajectory)
                    enc_start = max(0, valid_enc_len - max_enc_show)
                    enc_subset = slice(enc_start, valid_enc_len)
                    
                    # Select decoder subset (first M tokens - immediate predictions)
                    dec_subset = slice(0, max_dec_show)
                    
                    # Extract attention submatrix
                    attention_matrix = full_attention[dec_subset, enc_subset]
                    
                    # Get corresponding token labels
                    enc_labels = [f"{tok[:8]}" for tok in enc_tokens[enc_start:valid_enc_len]]
                    dec_labels = [f"{tok[:8]}" for tok in dec_tokens[:max_dec_show]]
                    
                    print(f"   🎯 Visualizing subset - Encoder: {len(enc_labels)} (last {max_enc_show}), Decoder: {len(dec_labels)} (first {max_dec_show})")
                    
                    # Create visualization
                    fig_width = max(10, len(enc_labels) * 0.6)
                    fig_height = max(6, len(dec_labels) * 0.4)
                    plt.figure(figsize=(fig_width, fig_height))
                    
                    # Create heatmap
                    sns.heatmap(attention_matrix, 
                                xticklabels=enc_labels, 
                                yticklabels=dec_labels,
                                cmap='Blues', 
                                cbar=True, 
                                square=False, 
                                linewidths=0.2,
                                annot=True if attention_matrix.size < 200 else False,
                                fmt='.2f',
                                cbar_kws={'label': 'Attention Weight'})
                    
                    plt.title(f'Cross-Attention Pattern - UID: {uid}\\n'
                             f'Recent Past ({max_enc_show} tokens) → Near Future ({max_dec_show} tokens)\\n'
                             f'Last Layer, Head 0')
                    plt.xlabel('← Encoder Tokens (Recent Past Trajectory)')
                    plt.ylabel('← Decoder Tokens (Future Predictions)')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    plt.show()
                    
                    # Print attention statistics
                    print(f"   📊 Attention Statistics:")
                    
                    # Calculate attention entropy (how focused the attention is)
                    entropies = []
                    for dec_pos in range(attention_matrix.shape[0]):
                        probs = attention_matrix[dec_pos]
                        probs = probs / (probs.sum() + 1e-8)
                        entropy = -np.sum(probs * np.log(probs + 1e-8))
                        entropies.append(entropy)
                    
                    avg_entropy = np.mean(entropies)
                    print(f"      Avg attention entropy: {avg_entropy:.3f} ({'Focused' if avg_entropy < 2.0 else 'Diffuse'})")
                    
                    # Most attended encoder positions
                    avg_attention = np.mean(attention_matrix, axis=0)
                    top_positions = np.argsort(avg_attention)[-3:][::-1]
                    
                    print(f"      🎯 Most attended recent encoder positions:")
                    for i, pos in enumerate(top_positions):
                        if pos < len(enc_labels):
                            token_str = enc_labels[pos]
                            actual_pos = enc_start + pos
                            print(f"         {i+1}. Pos {actual_pos}: {token_str} (attention: {avg_attention[pos]:.3f})")
                            
                else:
                    print("   ❌ No cross-attention found in model output.")
                    
        finally:
            # Restore original settings
            model.model.config.output_attentions = orig_att_setting
            
    except Exception as e:
        print(f"   ❌ Attention visualization failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main trajectory generation function from enriched Parquet data."""
    parser = argparse.ArgumentParser(description='Generate Trajectories from Enriched Parquet Data')
    parser.add_argument('--city', type=str, default='A',
                       choices=['A', 'B', 'C', 'D'],
                       help='City identifier (A, B, C, or D)')
    parser.add_argument('--checkpoint_path', type=str, 
                       default=None,
                       help='Path to model checkpoint (default: auto-generated based on city)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to enriched Parquet file (e.g., city_A_challengedata_enriched.parquet)')
    parser.add_argument('--target_days', type=str, default='69,75',
                       help='Target day range for generation (start,end)')
    parser.add_argument('--target_uids', type=str, default='1,10',
                       help='Target UID range (start,end) - optional')
    parser.add_argument('--max_encoder_steps', type=int, default=250,
                       help='Maximum number of encoder trajectory steps')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device for generation')
    parser.add_argument('--generation_strategy', type=str, default='sampling',
                       choices=['greedy', 'sampling', 'beam'],
                       help='Generation strategy')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Top-k for sampling strategy')
    parser.add_argument('--beam_width', type=int, default=5,
                       help='Beam width for beam search')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (1.0 = no change)')
    parser.add_argument('--save_comparison', action='store_true',
                       help='Save detailed comparison with ground truth')
    parser.add_argument('--snap_mode', type=str, default='none',
                       choices=['none', 'history', 'reachable'],
                       help='Snap mode for post-processing')
    parser.add_argument('--reachable_points', type=str, default=None,
                       help='Path to reachable points file')
    
    args = parser.parse_args()
    
    # Parse target days
    target_days = tuple(map(int, args.target_days.split(',')))
    if len(target_days) != 2:
        raise ValueError("target_days must be 'start,end' format")
    
    # Parse target UIDs if provided
    target_uids = None
    if args.target_uids:
        target_uids = tuple(map(int, args.target_uids.split(',')))
        if len(target_uids) != 2:
            raise ValueError("target_uids must be 'start,end' format")
    
    # Extract city from data path for output naming
    city = args.city
    
    # Auto-generate output filename if not provided
    if args.output_file is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        uid_suffix = f"_uids{target_uids[0]}-{target_uids[1]}" if target_uids else ""
        args.output_file = f"generated_trajectories_days{target_days[0]}-{target_days[1]}_{args.generation_strategy}_{timestamp}_city{city}{uid_suffix}.csv"
    
    # Get device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Optimized Trajectory Generation from Enriched Parquet Data")
    print(f"📍 Device: {device}")
    print(f"🎯 Strategy: {args.generation_strategy}")
    print(f"🏙️  City: {city}")
    print(f"📅 Target days: {target_days[0]} - {target_days[1]}")
    print(f"👥 Target UIDs: {target_uids if target_uids else 'All users'}")
    print(f"📏 Max encoder steps: {args.max_encoder_steps}")
    
    # Check data file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Enriched data file not found: {args.data_path}")
    
    # Load and prepare data
    generation_data = load_enriched_data(
        parquet_path=args.data_path,
        target_days=target_days,
        target_uids=target_uids,
        max_encoder_steps=args.max_encoder_steps
    )

    reachable_points = None
    if args.snap_mode == 'reachable':
        vocab_path = args.reachable_vocab or os.path.join('datasets', 'processed', 'vocabs', f'city_{city}_reachable_locations.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                data = json.load(f)
            locs = data.get('location_tokens', [])
            if isinstance(locs, list) and len(locs) > 0:
                reachable_points = np.array(locs, dtype=np.int32)
                print(f"✓ Loaded {reachable_points.shape[0]:,} reachable locations from {vocab_path}")
        else:
            print(f"⚠️ Reachable vocab not found at {vocab_path}; snap_mode ignored")


    
    if len(generation_data) == 0:
        print("❌ No users found for generation after filtering")
        return
    
    # read number_of clusters from pkl file
    metadata_file = os.path.join('datasets', 'processed', f'{city}_mixed_trajectory_metadata.pkl')
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
        cluster_info = metadata.get('cluster_info', {})
        num_clusters = cluster_info.get('num_clusters', 0)
        print(f"Loaded metadata with {num_clusters} clusters")
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join('checkpoints', 'trajectory_transformer', f'best_model_{city}.pt')
    model = load_model_from_checkpoint(args.checkpoint_path, device, num_clusters)
    
    # Generate trajectories
    start_time = time.time()
    results = generate_trajectories_from_data(
        model=model,
        generation_data=generation_data,
        device=device,
        generation_strategy=args.generation_strategy,
        top_k=args.top_k,
        beam_width=args.beam_width,
        temperature=args.temperature,
        snap_mode=args.snap_mode,
        reachable_points=reachable_points
    )
    total_time = time.time() - start_time
    
    # Save results
    if results:
        save_results_to_csv(results, args.output_file)
        
        if args.save_comparison:
            save_comparison_results(results, args.output_file)
    else:
        print("❌ No results to save")
    
    print(f"\n⏱️  Total pipeline time: {total_time:.2f}s")
    print(f"🏁 Generation completed successfully!")

if __name__ == '__main__':
    main()