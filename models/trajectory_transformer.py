import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch.nn.functional as F

class TrajectoryT5(nn.Module):
    """
    T5-based transformer for trajectory prediction with separate temporal and cluster embeddings.
    """
    
    def __init__(self, tokenizer, model_name: str = 't5-small', 
                 from_pretrained: bool = False,
                 model_path: Optional[str] = None,
                 label_smoothing_factor: float = 0.0,
                 t_vocab_size: int = 48,
                 dow_vocab_size: int = 7,
                 td_vocab_size: int = 7,
                 num_clusters: int = 100,  # Add cluster vocabulary size
                 ):
        """
        Initialize trajectory transformer with cluster embeddings.
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.num_clusters = num_clusters

        if from_pretrained:
            print("Initializing model from pretrained")
            if model_path is None:
                # from huggingface
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name, 
                    label_smoothing_factor=label_smoothing_factor
                )
                print(f"🔧 Loading pretrained model from {model_name}")
            else:
                print(f"🔧 Loading pretrained model from {model_path}")
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_path, 
                    label_smoothing_factor=label_smoothing_factor
                )
        else:
            # Always create from scratch in offline mode
            print("🔧 Creating T5 model from scratch (offline mode)")
        
            # Use default T5 configuration based on model size
            if 't5-small' in model_name:
                config = T5Config(
                    vocab_size=tokenizer.vocab_size,
                    d_model=512,
                    d_ff=2048,
                    d_kv=64,
                    num_heads=8,
                    num_layers=6,
                    num_decoder_layers=6,
                    dropout_rate=0.1,
                    layer_norm_epsilon=1e-6,
                    initializer_factor=1.0,
                    feed_forward_proj="relu",
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.end_token_id,
                    decoder_start_token_id=tokenizer.start_token_id,
                    label_smoothing_factor=label_smoothing_factor
                )
            elif 't5-base' in model_name:
                config = T5Config(
                    vocab_size=tokenizer.vocab_size,
                    d_model=768,
                    d_ff=3072,
                    d_kv=64,
                    num_heads=12,
                    num_layers=12,
                    num_decoder_layers=12,
                    dropout_rate=0.1,
                    layer_norm_epsilon=1e-6,
                    initializer_factor=1.0,
                    feed_forward_proj="relu",
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.end_token_id,
                    decoder_start_token_id=tokenizer.start_token_id,
                    label_smoothing_factor=label_smoothing_factor
                )
            else:  # Default to small
                config = T5Config(
                    vocab_size=tokenizer.vocab_size,
                    d_model=512,
                    d_ff=2048,
                    d_kv=64,
                    num_heads=8,
                    num_layers=6,
                    num_decoder_layers=6,
                    dropout_rate=0.1,
                    layer_norm_epsilon=1e-6,
                    initializer_factor=1.0,
                    feed_forward_proj="relu",
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.end_token_id,
                    decoder_start_token_id=tokenizer.start_token_id,
                    label_smoothing_factor=label_smoothing_factor
                )
            
            self.model = T5ForConditionalGeneration(config)
            print(f"✓ Created {model_name} model from scratch")
        
        # Get embedding dimension from T5 model
        self.embed_dim = self.model.config.d_model
        
        # Store vocab sizes and define padding indices
        self.t_vocab_size = t_vocab_size
        self.dow_vocab_size = dow_vocab_size
        self.td_vocab_size = td_vocab_size
        
        self.t_pad_idx = t_vocab_size
        self.dow_pad_idx = dow_vocab_size
        self.td_pad_idx = td_vocab_size
        self.cluster_pad_idx = num_clusters  # Padding index for clusters

        # Add temporal embedding layers with padding_idx
        self.t_embedding = nn.Embedding(self.t_vocab_size + 1, self.embed_dim, padding_idx=self.t_pad_idx)
        self.dow_embedding = nn.Embedding(self.dow_vocab_size + 1, self.embed_dim, padding_idx=self.dow_pad_idx)
        self.td_embedding = nn.Embedding(self.td_vocab_size + 1, self.embed_dim, padding_idx=self.td_pad_idx)
        
        # Add cluster embedding layer
        self.cluster_embedding = nn.Embedding(self.num_clusters + 1, self.embed_dim, padding_idx=self.cluster_pad_idx)
        
        # Initialize temporal and cluster embeddings
        nn.init.normal_(self.t_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dow_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.td_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cluster_embedding.weight, mean=0.0, std=0.02)

        # Zero out padding embeddings
        with torch.no_grad():
            self.t_embedding.weight[self.t_pad_idx].fill_(0)
            self.dow_embedding.weight[self.dow_pad_idx].fill_(0)
            self.td_embedding.weight[self.td_pad_idx].fill_(0)
            self.cluster_embedding.weight[self.cluster_pad_idx].fill_(0)
        
        print(f"Model initialized with vocab size: {self.model.config.vocab_size}")
        print(f"Added temporal embeddings: t({self.t_vocab_size+1}), dow({self.dow_vocab_size+1}), td({self.td_vocab_size+1})")
        print(f"Added cluster embedding: clusters({self.num_clusters+1})")
        print(f"Padding indices: t={self.t_pad_idx}, dow={self.dow_pad_idx}, td={self.td_pad_idx}, cluster={self.cluster_pad_idx}")
        
    def _adapt_embeddings(self):
        """Adapt pretrained embeddings to our vocabulary size."""
        old_vocab_size = self.model.config.vocab_size
        new_vocab_size = self.tokenizer.vocab_size
        
        if old_vocab_size != new_vocab_size:
            print(f"Adapting embeddings from {old_vocab_size} to {new_vocab_size}")
            
            # Resize embeddings
            self.model.resize_token_embeddings(new_vocab_size)
            
            # Initialize new tokens with small random values
            with torch.no_grad():
                if new_vocab_size > old_vocab_size:
                    # Initialize new tokens
                    new_embeddings = self.model.shared.weight[old_vocab_size:]
                    new_embeddings.normal_(mean=0.0, std=0.02)
        
        # Update config
        self.model.config.vocab_size = new_vocab_size
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.end_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.start_token_id
    
    def forward(self, encoder_input_ids: torch.Tensor, 
                encoder_temporal: torch.Tensor,
                encoder_attention_mask: torch.Tensor,
                decoder_input_ids: torch.Tensor,
                decoder_temporal: torch.Tensor,
                decoder_attention_mask: torch.Tensor,
                encoder_clusters: torch.Tensor,  # Add cluster inputs
                decoder_clusters: torch.Tensor,  # Add cluster inputs
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass with temporal and cluster embeddings.
        
        Args:
            encoder_input_ids: Coordinate tokens [batch, enc_seq_len]
            encoder_temporal: Temporal features [batch, enc_seq_len, 3] (t, dow, td)
            encoder_attention_mask: Encoder attention mask [batch, enc_seq_len]
            decoder_input_ids: Coordinate tokens [batch, dec_seq_len]
            decoder_temporal: Temporal features [batch, dec_seq_len, 3] (t, dow, td)
            decoder_attention_mask: Decoder attention mask [batch, dec_seq_len]
            encoder_clusters: Cluster indices [batch, enc_seq_len]
            decoder_clusters: Cluster indices [batch, dec_seq_len]
            labels: Target coordinate tokens [batch, target_len]
        """
        # Get coordinate embeddings from T5
        encoder_coord_embeds = self.model.shared(encoder_input_ids)
        decoder_coord_embeds = self.model.shared(decoder_input_ids)
        
        # Get temporal embeddings
        encoder_t_embeds = self.t_embedding(encoder_temporal[:, :, 0])
        encoder_dow_embeds = self.dow_embedding(encoder_temporal[:, :, 1])
        encoder_td_embeds = self.td_embedding(encoder_temporal[:, :, 2])
        
        decoder_t_embeds = self.t_embedding(decoder_temporal[:, :, 0])
        decoder_dow_embeds = self.dow_embedding(decoder_temporal[:, :, 1])
        decoder_td_embeds = self.td_embedding(decoder_temporal[:, :, 2])
        
        # Get cluster embeddings
        encoder_cluster_embeds = self.cluster_embedding(encoder_clusters)
        decoder_cluster_embeds = self.cluster_embedding(decoder_clusters)
        
        # Combine embeddings (coordinate + temporal + cluster)
        encoder_embeds = (encoder_coord_embeds + encoder_t_embeds + encoder_dow_embeds + 
                         encoder_td_embeds + encoder_cluster_embeds)
        decoder_embeds = (decoder_coord_embeds + decoder_t_embeds + decoder_dow_embeds + 
                         decoder_td_embeds + decoder_cluster_embeds)
        
        # Forward through T5 with custom embeddings
        return self.model(
            inputs_embeds=encoder_embeds,
            attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=decoder_embeds,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
    
    def prepare_batch_data(self, batch_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Vectorized batch prep: tokenizes coordinates, aligns temporal/cluster features, pads on CPU,
        then performs a single device transfer per tensor. Preserves current decoder alignment:
        [START, x1, y1, ..., xN, yN] ↔ [t1, t1, t1, ..., tN, tN].
        """
        device = next(self.parameters()).device
        batch_size = batch_data['batch_size']

        # Convenience handles
        enc_in = batch_data['encoder_input']    # [B, L_enc, 5] (floats from collate, will cast)
        dec_in = batch_data['decoder_input']    # [B, L_dec, 5]
        dec_tg = batch_data['decoder_target']   # [B, L_dec, 2]
        enc_mask = batch_data['encoder_mask']   # [B, L_enc] bool
        dec_mask = batch_data['decoder_mask']   # [B, L_dec] bool
        clusters = batch_data['cluster_idx']    # [B]

        # Lists to hold ragged per-sample sequences
        enc_ids_list, enc_temp_list, enc_clu_list = [], [], []
        dec_ids_list, dec_temp_list, dec_clu_list = [], [], []
        labels_list = []

        start_id = self.tokenizer.start_token_id
        end_id = self.tokenizer.end_token_id
        pad_id = self.tokenizer.pad_token_id
        x_off = self.tokenizer.x_offset
        y_off = self.tokenizer.y_offset

        for i in range(batch_size):
            cluster_idx = int(clusters[i].item())

            # ------- Encoder (vectorized per sample) -------
            e_len = int(enc_mask[i].sum().item())
            if e_len > 0:
                enc_slice = enc_in[i, :e_len].long()  # [e_len,5]
                ex = enc_slice[:, 0]
                ey = enc_slice[:, 1]
                # Optionally filter residual paddings (safety)
                valid_e = (ex != -999) & (ey != -999)
                if valid_e.any():
                    ex = ex[valid_e]
                    ey = ey[valid_e]
                    et = enc_slice[valid_e, 2:5]  # [n,3]

                    x_tokens = x_off + ex
                    y_tokens = y_off + ey
                    enc_tokens = torch.stack([x_tokens, y_tokens], dim=1).reshape(-1)  # [2n]

                    enc_temporal = et.repeat_interleave(2, dim=0)  # [2n,3]
                    enc_clusters = torch.full((enc_tokens.numel(),), cluster_idx, dtype=torch.long)

                    enc_ids_list.append(enc_tokens)
                    enc_temp_list.append(enc_temporal)
                    enc_clu_list.append(enc_clusters)
                else:
                    enc_ids_list.append(torch.empty(0, dtype=torch.long))
                    enc_temp_list.append(torch.empty(0, 3, dtype=torch.long))
                    enc_clu_list.append(torch.empty(0, dtype=torch.long))
            else:
                enc_ids_list.append(torch.empty(0, dtype=torch.long))
                enc_temp_list.append(torch.empty(0, 3, dtype=torch.long))
                enc_clu_list.append(torch.empty(0, dtype=torch.long))

            # ------- Decoder (vectorized per sample) -------
            d_len = int(dec_mask[i].sum().item())
            if d_len > 0:
                d_in = dec_in[i, :d_len].long()    # [d_len,5]
                d_tg = dec_tg[i, :d_len].long()    # [d_len,2]

                vx = d_tg[:, 0]
                vy = d_tg[:, 1]
                valid_d = (vx != -999) & (vy != -999)
                if valid_d.any():
                    d_in = d_in[valid_d]          # [n,5]
                    d_tg = d_tg[valid_d]          # [n,2]
                    n = d_in.size(0)

                    x_tokens = x_off + d_tg[:, 0]
                    y_tokens = y_off + d_tg[:, 1]
                    xy_tokens = torch.stack([x_tokens, y_tokens], dim=1).reshape(-1)  # [2n]

                    # Decoder input ids: [START] + xy tokens
                    dec_tokens = torch.empty(1 + xy_tokens.numel(), dtype=torch.long)
                    dec_tokens[0] = start_id
                    dec_tokens[1:] = xy_tokens

                    # Temporal alignment: START uses t1; each x_k/y_k uses t_k; last t repeated for y_N
                    future_t = d_in[:, 2:5]                     # [n,3]
                    rep_t = future_t.repeat_interleave(2, 0)    # [2n,3]
                    start_t = future_t[0].unsqueeze(0)          # [1,3]
                    dec_temporal = torch.cat([start_t, rep_t], dim=0)  # [1+2n,3]

                    dec_clusters = torch.full((dec_tokens.numel(),), cluster_idx, dtype=torch.long)

                    # Labels: tokens for targets + END
                    labels = torch.empty(xy_tokens.numel() + 1, dtype=torch.long)
                    labels[:-1] = xy_tokens
                    labels[-1] = end_id

                    dec_ids_list.append(dec_tokens)
                    dec_temp_list.append(dec_temporal)
                    dec_clu_list.append(dec_clusters)
                    labels_list.append(labels)
                else:
                    # No valid targets; still provide minimal START and label END
                    dec_ids_list.append(torch.tensor([start_id], dtype=torch.long))
                    dec_temp_list.append(torch.tensor(
                        [[self.t_pad_idx, self.dow_pad_idx, self.td_pad_idx]], dtype=torch.long))
                    dec_clu_list.append(torch.full((1,), cluster_idx, dtype=torch.long))
                    labels_list.append(torch.tensor([end_id], dtype=torch.long))
            else:
                dec_ids_list.append(torch.tensor([start_id], dtype=torch.long))
                dec_temp_list.append(torch.tensor(
                    [[self.t_pad_idx, self.dow_pad_idx, self.td_pad_idx]], dtype=torch.long))
                dec_clu_list.append(torch.full((1,), cluster_idx, dtype=torch.long))
                labels_list.append(torch.tensor([end_id], dtype=torch.long))

        # ------- Pad on CPU, then single transfer to device -------
        max_enc = max((t.numel() for t in enc_ids_list), default=1)
        max_dec = max((t.numel() for t in dec_ids_list), default=1)
        max_lab = max((t.numel() for t in labels_list), default=1)

        # CPU tensors
        enc_ids = torch.full((batch_size, max_enc), pad_id, dtype=torch.long)
        enc_temp = torch.full((batch_size, max_enc, 3), 0, dtype=torch.long)
        enc_temp[..., 0] = self.t_pad_idx
        enc_temp[..., 1] = self.dow_pad_idx
        enc_temp[..., 2] = self.td_pad_idx
        enc_clu = torch.full((batch_size, max_enc), self.cluster_pad_idx, dtype=torch.long)

        dec_ids = torch.full((batch_size, max_dec), pad_id, dtype=torch.long)
        dec_temp = torch.full((batch_size, max_dec, 3), 0, dtype=torch.long)
        dec_temp[..., 0] = self.t_pad_idx
        dec_temp[..., 1] = self.dow_pad_idx
        dec_temp[..., 2] = self.td_pad_idx
        dec_clu = torch.full((batch_size, max_dec), self.cluster_pad_idx, dtype=torch.long)

        labels = torch.full((batch_size, max_lab), -100, dtype=torch.long)

        for i in range(batch_size):
            # Encoder fill
            if enc_ids_list[i].numel() > 0:
                L = enc_ids_list[i].numel()
                enc_ids[i, :L] = enc_ids_list[i]
                enc_temp[i, :L] = enc_temp_list[i]
                enc_clu[i, :L] = enc_clu_list[i]
            # Decoder fill
            if dec_ids_list[i].numel() > 0:
                L = dec_ids_list[i].numel()
                dec_ids[i, :L] = dec_ids_list[i]
                dec_temp[i, :L] = dec_temp_list[i]
                dec_clu[i, :L] = dec_clu_list[i]
            # Labels fill
            if labels_list[i].numel() > 0:
                L = labels_list[i].numel()
                labels[i, :L] = labels_list[i]

        enc_attn = (enc_ids != pad_id).long()
        dec_attn = (dec_ids != pad_id).long()
        # print(enc_ids.shape, dec_ids.shape, labels.shape) #debug

        # Single transfers
        return {
            'encoder_input_ids': enc_ids.to(device, non_blocking=True),
            'encoder_temporal': enc_temp.to(device, non_blocking=True),
            'encoder_clusters': enc_clu.to(device, non_blocking=True),
            'encoder_attention_mask': enc_attn.to(device, non_blocking=True),
            'decoder_input_ids': dec_ids.to(device, non_blocking=True),
            'decoder_temporal': dec_temp.to(device, non_blocking=True),
            'decoder_clusters': dec_clu.to(device, non_blocking=True),
            'decoder_attention_mask': dec_attn.to(device, non_blocking=True),
            'labels': labels.to(device, non_blocking=True),
        }
    
    # greedy generation
    def generate_trajectory(self, 
                          batch_past_coords: List[List[Tuple[int, int]]],
                          batch_past_temporal: List[List[Tuple[int, int, int]]],
                          batch_future_temporal: List[List[Tuple[int, int, int]]],
                          batch_cluster_indices: List[int],  # Add cluster indices as argument
                          device: torch.device) -> List[List[Tuple[int, int]]]:
        """
        Generate trajectory coordinates for a batch.
        Optimized: vectorized encoder prep + cached decoder generation.
        """
        self.eval()
        batch_size = len(batch_past_coords)

        # Tokenize and pad encoder inputs (vectorized)
        encoder_token_sequences, encoder_temporal_sequences = [], []
        for past_coords, past_temporal in zip(batch_past_coords, batch_past_temporal):
            tokens, temporal = [], []
            for (x, y), (t, dow, td) in zip(past_coords, past_temporal):
                tokens.extend(self.tokenizer.encode_coordinates(x, y))
                temporal.extend([[t, dow, td], [t, dow, td]])
            encoder_token_sequences.append(tokens)
            encoder_temporal_sequences.append(temporal)

        max_len = max((len(s) for s in encoder_token_sequences), default=0)
        encoder_input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        encoder_temporal = torch.full((batch_size, max_len, 3), 0, dtype=torch.long, device=device)
        encoder_temporal[..., 0] = self.t_pad_idx
        encoder_temporal[..., 1] = self.dow_pad_idx
        encoder_temporal[..., 2] = self.td_pad_idx
        # Correctly create encoder_clusters tensor
        encoder_clusters = torch.full((batch_size, max_len), self.cluster_pad_idx, dtype=torch.long, device=device)

        for i, (tokens, temporal) in enumerate(zip(encoder_token_sequences, encoder_temporal_sequences)):
            if tokens:
                encoder_input_ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
                encoder_temporal[i, :len(temporal)] = torch.tensor(temporal, dtype=torch.long, device=device)
                # Assign the real cluster index to all tokens for this user
                encoder_clusters[i, :len(tokens)] = batch_cluster_indices[i]

        encoder_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id)

        # This was the bug: it was creating a list of padding indices.
        # batch_cluster_indices = [self.cluster_pad_idx] * batch_size

        return self._generate_batch_autoregressive(
            encoder_input_ids, encoder_temporal, encoder_clusters, encoder_attention_mask,
            batch_future_temporal, batch_cluster_indices, device
        )
    
    def _generate_batch_autoregressive(self,
                                       encoder_input_ids: torch.Tensor,
                                       encoder_temporal: torch.Tensor,
                                       encoder_clusters: torch.Tensor,
                                       encoder_attention_mask: torch.Tensor,
                                       batch_future_temporal: List[List[Tuple[int, int, int]]],
                                       batch_cluster_indices: List[int],
                                       device: torch.device) -> List[List[Tuple[int, int]]]:
        """
        Faster greedy generation with x,y pattern enforcement and caching.
        """
        self.eval()
        start_id = self.tokenizer.start_token_id
        end_id = self.tokenizer.end_token_id
        batch_size = encoder_input_ids.size(0)

        # Precompute encoder embeddings and run encoder once
        with torch.inference_mode():
            enc_coord_embeds = self.model.shared(encoder_input_ids)
            enc_t = self.t_embedding(encoder_temporal[:, :, 0])
            enc_dow = self.dow_embedding(encoder_temporal[:, :, 1])
            enc_td = self.td_embedding(encoder_temporal[:, :, 2])
            enc_clu = self.cluster_embedding(encoder_clusters)
            encoder_embeds = enc_coord_embeds + enc_t + enc_dow + enc_td + enc_clu
            #without cluster embedding
            # encoder_embeds = enc_coord_embeds + enc_t + enc_dow + enc_td

            encoder_outputs = self.model.encoder(
                inputs_embeds=encoder_embeds,
                attention_mask=encoder_attention_mask,
                return_dict=True
            )

        # Initialize generation state
        batch_decoder_tokens = [[start_id] for _ in range(batch_size)]
        batch_predicted_coords = [[] for _ in range(batch_size)]
        is_finished = [False] * batch_size
        expecting_x_token = [True] * batch_size  # After START, expect x token
        past_key_values = None

        def temporal_index(tokens_len: int, fut_len: int) -> int:
            pos = tokens_len - 1
            return min(pos // 2, max(fut_len - 1, 0))

        max_future_len = max((len(fut) for fut in batch_future_temporal), default=0)
        max_steps = max_future_len * 2

        with torch.inference_mode():
            for _ in range(max_steps):
                if all(is_finished):
                    break

                # Build one-step decoder inputs for entire batch
                last_token_ids, t_list, dow_list, td_list, clu_list = [], [], [], [], []

                for i in range(batch_size):
                    if is_finished[i]:
                        last_token_ids.append(end_id)
                        fut_len = len(batch_future_temporal[i]) if batch_future_temporal else 0
                        step_idx = max(0, fut_len - 1)
                    else:
                        last_token_ids.append(batch_decoder_tokens[i][-1])
                        fut_len = len(batch_future_temporal[i]) if batch_future_temporal else 0
                        step_idx = temporal_index(len(batch_decoder_tokens[i]), fut_len)

                    if fut_len > 0:
                        t, dow, td = batch_future_temporal[i][step_idx]
                    else:
                        t, dow, td = self.t_pad_idx, self.dow_pad_idx, self.td_pad_idx

                    t_list.append(t); dow_list.append(dow); td_list.append(td)
                    clu_list.append(batch_cluster_indices[i] if batch_cluster_indices else self.cluster_pad_idx)

                last_token_ids = torch.tensor(last_token_ids, dtype=torch.long, device=device)
                t_idx = torch.tensor(t_list, dtype=torch.long, device=device)
                dow_idx = torch.tensor(dow_list, dtype=torch.long, device=device)
                td_idx = torch.tensor(td_list, dtype=torch.long, device=device)
                clu_idx = torch.tensor(clu_list, dtype=torch.long, device=device)

                dec_coord = self.model.shared(last_token_ids).unsqueeze(1)
                dec_t = self.t_embedding(t_idx).unsqueeze(1)
                dec_dow = self.dow_embedding(dow_idx).unsqueeze(1)
                dec_td = self.td_embedding(td_idx).unsqueeze(1)
                dec_clu = self.cluster_embedding(clu_idx).unsqueeze(1)
                dec_embeds = dec_coord + dec_t + dec_dow + dec_td + dec_clu
                #without cluster embedding
                # dec_embeds = dec_coord + dec_t + dec_dow + dec_td

                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=dec_embeds,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values

                # Process each sample with pattern enforcement
                raw_logits = outputs.logits[:, -1, :]  # [B,V]
                
                for i in range(batch_size):
                    if not is_finished[i]:
                        # Use universal validation function for greedy selection
                        predicted_token = self._enforce_xy_pattern_validation(
                            logits=raw_logits[i],
                            expecting_x_token=expecting_x_token[i],
                            use_sampling=False  # Greedy
                        )
                        
                        # Process the token and update state
                        is_finished[i] = self._process_generated_token(
                            sample_idx=i,
                            predicted_token=predicted_token,
                            batch_decoder_tokens=batch_decoder_tokens,
                            batch_predicted_coords=batch_predicted_coords,
                            expecting_x_token=expecting_x_token,
                            batch_future_temporal=batch_future_temporal
                        )

        return batch_predicted_coords

    def generate_trajectory_with_sampling(self, 
                                          batch_past_coords: List[List[Tuple[int, int]]],
                                          batch_past_temporal: List[List[Tuple[int, int, int]]],
                                          batch_future_temporal: List[List[Tuple[int, int, int]]],
                                          batch_cluster_indices: List[int], # Add cluster indices as argument
                                          device: torch.device,
                                          top_k: int = 50,
                                          temperature: float = 1.0) -> List[List[Tuple[int, int]]]:
        """
        Generate trajectory coordinates using top-k sampling (optimized).
        - Vectorized encoder prep
        - Cached encoder states
        - Decoder uses use_cache + past_key_values, feeding only the last token each step
        """
        self.eval()
        batch_size = len(batch_past_coords)

        # Vectorized encoder tokenization
        encoder_token_sequences, encoder_temporal_sequences = [], []
        for past_coords, past_temporal in zip(batch_past_coords, batch_past_temporal):
            tokens, temporal = [], []
            for (x, y), (t, dow, td) in zip(past_coords, past_temporal):
                tokens.extend(self.tokenizer.encode_coordinates(x, y))
                temporal.extend([[t, dow, td], [t, dow, td]])
            encoder_token_sequences.append(tokens)
            encoder_temporal_sequences.append(temporal)

        max_len = max((len(s) for s in encoder_token_sequences), default=0)
        encoder_input_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        encoder_temporal = torch.full((batch_size, max_len, 3), 0, dtype=torch.long, device=device)
        encoder_temporal[..., 0] = self.t_pad_idx
        encoder_temporal[..., 1] = self.dow_pad_idx
        encoder_temporal[..., 2] = self.td_pad_idx
        encoder_clusters = torch.full((batch_size, max_len), self.cluster_pad_idx, dtype=torch.long, device=device)

        for i, (tokens, temporal) in enumerate(zip(encoder_token_sequences, encoder_temporal_sequences)):
            if tokens:
                encoder_input_ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
                encoder_temporal[i, :len(temporal)] = torch.tensor(temporal, dtype=torch.long, device=device)
                # Assign the real cluster index to all tokens for this user
                encoder_clusters[i, :len(tokens)] = batch_cluster_indices[i]

        encoder_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id)

        # This was the bug: it was creating a list of padding indices.
        # batch_cluster_indices = [self.cluster_pad_idx] * batch_size

        return self._generate_batch_autoregressive_with_sampling(
            encoder_input_ids,
            encoder_temporal,
            encoder_clusters,
            encoder_attention_mask,
            batch_future_temporal,
            batch_cluster_indices,
            device,
            top_k,
            temperature
        )

    def _generate_batch_autoregressive_with_sampling(self,
                                                      encoder_input_ids: torch.Tensor,
                                                      encoder_temporal: torch.Tensor,
                                                      encoder_clusters: torch.Tensor,
                                                      encoder_attention_mask: torch.Tensor,
                                                      batch_future_temporal: List[List[Tuple[int, int, int]]],
                                                      batch_cluster_indices: List[int],
                                                      device: torch.device,
                                                      top_k: int,
                                                      temperature: float) -> List[List[Tuple[int, int]]]:
        """
        Optimized top-k sampling with x,y pattern enforcement.
        """
        self.eval()
        start_id = self.tokenizer.start_token_id
        end_id = self.tokenizer.end_token_id
        batch_size = encoder_input_ids.size(0)

        # Encoder pass once
        with torch.inference_mode():
            enc_coord = self.model.shared(encoder_input_ids)
            enc_t = self.t_embedding(encoder_temporal[:, :, 0])
            enc_dow = self.dow_embedding(encoder_temporal[:, :, 1])
            enc_td = self.td_embedding(encoder_temporal[:, :, 2])
            enc_clu = self.cluster_embedding(encoder_clusters)
            encoder_embeds = enc_coord + enc_t + enc_dow + enc_td + enc_clu
            #without cluster embedding
            # encoder_embeds = enc_coord + enc_t + enc_dow + enc_td
            encoder_outputs = self.model.encoder(
                inputs_embeds=encoder_embeds,
                attention_mask=encoder_attention_mask,
                return_dict=True
            )

        # Init state
        batch_decoder_tokens = [[start_id] for _ in range(batch_size)]
        batch_predicted_coords = [[] for _ in range(batch_size)]
        is_finished = [False] * batch_size
        expecting_x_token = [True] * batch_size  # After START, expect x token
        past_key_values = None

        def temporal_index(tokens_len: int, fut_len: int) -> int:
            pos = tokens_len - 1
            return min(pos // 2, max(fut_len - 1, 0))

        max_future_len = max((len(fut) for fut in batch_future_temporal), default=0)
        max_steps = max_future_len * 2

        with torch.inference_mode():
            for _ in range(max_steps):
                if all(is_finished):
                    break

                # Build 1-step embeddings for the whole batch
                last_ids, t_list, dow_list, td_list, clu_list = [], [], [], [], []
                for i in range(batch_size):
                    if is_finished[i]:
                        last_ids.append(end_id)
                        fut_len = len(batch_future_temporal[i]) if batch_future_temporal else 0
                        step_idx = max(0, fut_len - 1)
                    else:
                        last_ids.append(batch_decoder_tokens[i][-1])
                        fut_len = len(batch_future_temporal[i]) if batch_future_temporal else 0
                        step_idx = temporal_index(len(batch_decoder_tokens[i]), fut_len)

                    if fut_len > 0:
                        t, dow, td = batch_future_temporal[i][step_idx]
                    else:
                        t, dow, td = self.t_pad_idx, self.dow_pad_idx, self.td_pad_idx

                    t_list.append(t); dow_list.append(dow); td_list.append(td)
                    clu_list.append(batch_cluster_indices[i] if batch_cluster_indices else self.cluster_pad_idx)

                last_ids = torch.tensor(last_ids, dtype=torch.long, device=device)
                t_idx = torch.tensor(t_list, dtype=torch.long, device=device)
                dow_idx = torch.tensor(dow_list, dtype=torch.long, device=device)
                td_idx = torch.tensor(td_list, dtype=torch.long, device=device)
                clu_idx = torch.tensor(clu_list, dtype=torch.long, device=device)

                dec_coord = self.model.shared(last_ids).unsqueeze(1)
                dec_t = self.t_embedding(t_idx).unsqueeze(1)
                dec_dow = self.dow_embedding(dow_idx).unsqueeze(1)
                dec_td = self.td_embedding(td_idx).unsqueeze(1)
                dec_clu = self.cluster_embedding(clu_idx).unsqueeze(1)
                dec_embeds = dec_coord + dec_t + dec_dow + dec_td + dec_clu
                #without cluster embedding
                # dec_embeds = dec_coord + dec_t + dec_dow + dec_td
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    decoder_inputs_embeds=dec_embeds,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values

                # Process each sample with pattern enforcement and sampling
                raw_logits = outputs.logits[:, -1, :]  # [B,V]
                
                for i in range(batch_size):
                    if not is_finished[i]:
                        # Use universal validation function for sampling
                        predicted_token = self._enforce_xy_pattern_validation(
                            logits=raw_logits[i],
                            expecting_x_token=expecting_x_token[i],
                            top_k=top_k,
                            temperature=temperature,
                            use_sampling=True  # Sampling
                        )
                        
                        # Process the token and update state
                        is_finished[i] = self._process_generated_token(
                            sample_idx=i,
                            predicted_token=predicted_token,
                            batch_decoder_tokens=batch_decoder_tokens,
                            batch_predicted_coords=batch_predicted_coords,
                            expecting_x_token=expecting_x_token,
                            batch_future_temporal=batch_future_temporal
                        )

        return batch_predicted_coords

    def generate_trajectory_with_beam_search(self,
                                            batch_past_coords: List[List[Tuple[int, int]]],
                                            batch_past_temporal: List[List[Tuple[int, int, int]]],
                                            batch_future_temporal: List[List[Tuple[int, int, int]]],
                                            batch_cluster_indices: List[int],
                                            device: torch.device,
                                            beam_width: int = 5) -> List[List[Tuple[int, int]]]:
        """
        Generate trajectory coordinates using beam search.
        Processes one sample at a time. Optimized with cached encoder outputs
        and incremental decoder with past_key_values.
        """
        if len(batch_past_coords) > 1:
            print("⚠️ Beam search is processing samples one by one.")
        
        all_predictions = []
        self.eval()

        for i in range(len(batch_past_coords)):
            # Prepare single-sample inputs
            past_coords = batch_past_coords[i]
            past_temporal = batch_past_temporal[i]
            future_temporal = batch_future_temporal[i]
            cluster_idx = batch_cluster_indices[i]

            # Tokenize encoder (vectorized for single sample)
            tokens, temporal = [], []
            for (x, y), (t, dow, td) in zip(past_coords, past_temporal):
                tokens.extend(self.tokenizer.encode_coordinates(x, y))
                temporal.extend([[t, dow, td], [t, dow, td]])
            
            max_len = len(tokens)
            if max_len == 0:
                all_predictions.append([])
                continue

            encoder_input_ids = torch.tensor([tokens], dtype=torch.long, device=device)             # [1, L]
            encoder_temporal = torch.full((1, max_len, 3), 0, dtype=torch.long, device=device)     # [1, L, 3]
            encoder_temporal[..., 0] = self.t_pad_idx
            encoder_temporal[..., 1] = self.dow_pad_idx
            encoder_temporal[..., 2] = self.td_pad_idx
            encoder_temporal[0, :len(temporal)] = torch.tensor(temporal, dtype=torch.long, device=device)

            encoder_clusters = torch.full((1, max_len), self.cluster_pad_idx, dtype=torch.long, device=device)
            encoder_clusters[0, :len(tokens)] = cluster_idx
            encoder_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id)

            # Build encoder embeddings and run encoder once
            with torch.inference_mode():
                enc_coord = self.model.shared(encoder_input_ids)
                enc_t = self.t_embedding(encoder_temporal[:, :, 0])
                enc_dow = self.dow_embedding(encoder_temporal[:, :, 1])
                enc_td = self.td_embedding(encoder_temporal[:, :, 2])
                enc_clu = self.cluster_embedding(encoder_clusters)
                encoder_embeds = enc_coord + enc_t + enc_dow + enc_td + enc_clu
                encoder_outputs = self.model.encoder(
                    inputs_embeds=encoder_embeds,
                    attention_mask=encoder_attention_mask,
                    return_dict=True
                )

            # Optimized single-sample beam search using cached encoder outputs
            predicted_coords = self._generate_autoregressive_with_beam_search(
                encoder_outputs=encoder_outputs,
                future_temporal=future_temporal,
                cluster_idx=cluster_idx,
                device=device,
                beam_width=beam_width
            )
            all_predictions.append(predicted_coords)

        return all_predictions

    def _generate_autoregressive_with_beam_search(self,
                                                encoder_outputs,
                                                future_temporal: List[Tuple[int, int, int]],
                                                cluster_idx: int,
                                                device: torch.device,
                                                beam_width: int) -> List[Tuple[int, int]]:
        """
        Optimized beam search with x,y pattern enforcement.
        """
        self.eval()
        start_id = self.tokenizer.start_token_id
        end_id = self.tokenizer.end_token_id

        max_steps = len(future_temporal) * 2
        if max_steps == 0:
            return []

        # Each beam: (seq, score, finished, past_key_values, expecting_x_token)
        beams = [([start_id], 0.0, False, None, True)]  # Start expecting x token

        def step_temporal_index(tokens_len: int) -> int:
            pos = tokens_len - 1
            return min(pos // 2, max(len(future_temporal) - 1, 0))

        with torch.inference_mode():
            for _ in range(max_steps):
                if all(f for _, _, f, _, _ in beams):
                    break

                candidates = []

                for seq, score, finished, pkv, expecting_x in beams:
                    if finished:
                        candidates.append((seq, score, True, pkv, expecting_x))
                        continue

                    # Prepare decoder embedding for last token
                    last_token_id = seq[-1]
                    step_idx = step_temporal_index(len(seq))
                    if len(future_temporal) > 0:
                        t, dow, td = future_temporal[step_idx]
                    else:
                        t, dow, td = self.t_pad_idx, self.dow_pad_idx, self.td_pad_idx

                    last_ids = torch.tensor([last_token_id], dtype=torch.long, device=device)
                    t_idx = torch.tensor([t], dtype=torch.long, device=device)
                    dow_idx = torch.tensor([dow], dtype=torch.long, device=device)
                    td_idx = torch.tensor([td], dtype=torch.long, device=device)
                    clu_idx = torch.tensor([cluster_idx], dtype=torch.long, device=device)

                    dec_coord = self.model.shared(last_ids).unsqueeze(1)
                    dec_t = self.t_embedding(t_idx).unsqueeze(1)
                    dec_dow = self.dow_embedding(dow_idx).unsqueeze(1)
                    dec_td = self.td_embedding(td_idx).unsqueeze(1)
                    dec_clu = self.cluster_embedding(clu_idx).unsqueeze(1)
                    dec_embeds = dec_coord + dec_t + dec_dow + dec_td + dec_clu
                    outputs = self.model(
                        encoder_outputs=encoder_outputs,
                        decoder_inputs_embeds=dec_embeds,
                        use_cache=True,
                        past_key_values=pkv,
                        return_dict=True
                    )
                    next_pkv = outputs.past_key_values
                    logits = outputs.logits[0, -1, :]
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Create a copy of logits for finding top-k, while preserving original for scoring
                    search_logits = logits.clone()

                    # Get top beam_width valid tokens using pattern enforcement
                    for _ in range(beam_width):
                        predicted_token = self._enforce_xy_pattern_validation(
                            logits=search_logits, # Use the mutable copy
                            expecting_x_token=expecting_x,
                            use_sampling=False
                        )
                        
                        # Calculate score from original, unmodified log_probs
                        token_score = log_probs[predicted_token].item()
                        new_score = score + token_score
                        new_seq = seq + [predicted_token]
                        
                        # Update expectation for next token
                        new_expecting_x = not expecting_x
                        
                        # Check if finished (completed enough coordinate pairs)
                        finished_now = False
                        if not expecting_x:  # Just generated y token
                            coords_so_far = len(self.tokenizer.decode_coordinates(new_seq[1:]))
                            if coords_so_far >= len(future_temporal):
                                finished_now = True

                        candidates.append((new_seq, new_score, finished_now, next_pkv, new_expecting_x))
                        
                        # Mask this token for next iteration to get diversity
                        search_logits[predicted_token] = -float('inf')

                # Select top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]

        # Choose best beam
        best_seq, _, _, _, _ = beams[0]
        predicted_tokens = best_seq[1:]  # drop START
        if predicted_tokens and predicted_tokens[-1] == end_id:
            predicted_tokens = predicted_tokens[:-1]

        coord_pairs = []
        for i in range(0, len(predicted_tokens), 2):
            if i + 1 < len(predicted_tokens):
                x_tok, y_tok = predicted_tokens[i], predicted_tokens[i + 1]
                coords = self.tokenizer.decode_coordinates([x_tok, y_tok])
                if coords:
                    coord_pairs.append(coords[0])

        return coord_pairs


    def _get_model_predictions_for_sampling(self, encoder_data, decoder_context, max_steps):
        """Generate model predictions for scheduled sampling."""
        with torch.inference_mode():
            # Run inference to get model's current predictions
            # This is a simplified version of generate_trajectory for training
            predictions = []
            decoder_tokens = [self.tokenizer.start_token_id]
            
            for step in range(max_steps):
                outputs = self.forward(
                    encoder_input_ids=encoder_data['input_ids'],
                    encoder_temporal=encoder_data['temporal'],
                    encoder_clusters=encoder_data['clusters'], # Pass clusters
                    encoder_attention_mask=encoder_data['attention_mask'],
                    decoder_input_ids=torch.tensor([decoder_tokens]).to(self.device),
                    decoder_temporal=decoder_context[:step+1],
                    decoder_clusters=torch.full((1, len(decoder_context[:step+1])), self.cluster_pad_idx, dtype=torch.long).to(self.device), # Pad cluster for decoder
                    decoder_attention_mask=torch.ones(len(decoder_tokens)).to(self.device)
                )
                
                # Get next x,y token predictions
                next_logits = outputs.logits[0, -1, :]
                x_token = torch.argmax(next_logits).item()
                decoder_tokens.append(x_token)
                
                # Similar for y token...
                
            return predictions


    def _enforce_xy_pattern_validation(self, logits: torch.Tensor, expecting_x_token: bool, 
                                      top_k: int = 0, temperature: float = 1.0, 
                                      use_sampling: bool = False) -> int:
        """
        Universal validation function for enforcing x,y,x,y token pattern.
        
        Args:
            logits: Raw logits from model [vocab_size]
            expecting_x_token: True if we expect x token, False if we expect y token
            top_k: For sampling methods, limit to top-k tokens (0 = no limit)
            temperature: Temperature scaling for sampling
            use_sampling: If True, use sampling; if False, use greedy (argmax)
            
        Returns:
            token_id: Valid token ID that respects the x,y pattern
        """
        # Get coordinate token ranges
        x_range = range(self.tokenizer.token_to_id[f'x_{self.tokenizer.x_range[0]}'], 
                       self.tokenizer.token_to_id[f'x_{self.tokenizer.x_range[1]}'] + 1)
        y_range = range(self.tokenizer.token_to_id[f'y_{self.tokenizer.y_range[0]}'], 
                       self.tokenizer.token_to_id[f'y_{self.tokenizer.y_range[1]}'] + 1)
        
        def is_x_token(token_id: int) -> bool:
            return token_id in x_range
            
        def is_y_token(token_id: int) -> bool:
            return token_id in y_range
        
        # Mask invalid tokens based on what we expect
        masked_logits = logits.clone()
        
        if expecting_x_token:
            # We expect x token, mask all y tokens
            masked_logits[y_range] = -float('inf')
        else:
            # We expect y token, mask all x tokens
            masked_logits[x_range] = -float('inf')

        # also mask the pad start and end and unknown tokens
        masked_logits[self.tokenizer.pad_token_id] = -float('inf')
        masked_logits[self.tokenizer.start_token_id] = -float('inf')
        masked_logits[self.tokenizer.end_token_id] = -float('inf')
        masked_logits[self.tokenizer.unk_token_id] = -float('inf')
        
        # Apply temperature scaling
        scaled_logits = masked_logits / max(temperature, 1e-6)
        
        # Generate token based on method
        if use_sampling:
            # Top-k sampling
            if top_k > 0 and top_k < scaled_logits.size(-1):
                top_vals, top_idx = torch.topk(scaled_logits, top_k)
                probs = torch.softmax(top_vals, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1).item()
                predicted_token = top_idx[sampled_idx].item()
            else:
                # Standard sampling
                probs = torch.softmax(scaled_logits, dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
        else:
            # Greedy selection
            predicted_token = torch.argmax(scaled_logits).item()
        
        # Safety validation (extra check)
        if expecting_x_token and not is_x_token(predicted_token):
            print(f"Warning: Expected x token but got {predicted_token}, forcing to valid x token")
            # Force to nearest valid x token as fallback
            predicted_token = min(x_range, key=lambda x: abs(x - predicted_token))
        elif not expecting_x_token and not is_y_token(predicted_token):
            print(f"Warning: Expected y token but got {predicted_token}, forcing to valid y token")
            # Force to nearest valid y token as fallback
            predicted_token = min(y_range, key=lambda y: abs(y - predicted_token))
        
        return predicted_token

    def _process_generated_token(self, sample_idx: int, predicted_token: int, 
                               batch_decoder_tokens: List[List[int]], 
                               batch_predicted_coords: List[List[Tuple[int, int]]], 
                               expecting_x_token: List[bool], 
                               batch_future_temporal: List[List[Tuple[int, int, int]]]) -> bool:
        """
        Universal function to process a generated token and update state.
        
        Args:
            sample_idx: Index of the sample in the batch
            predicted_token: The generated token ID
            batch_decoder_tokens: List of token sequences for each sample
            batch_predicted_coords: List of coordinate sequences for each sample
            expecting_x_token: List tracking what token type is expected next
            batch_future_temporal: Future temporal sequences for length checking
            
        Returns:
            is_finished: True if this sample has completed generation
        """
        # Add token to sequence
        batch_decoder_tokens[sample_idx].append(predicted_token)
        
        # Update expectation and check for coordinate completion
        if expecting_x_token[sample_idx]:
            # We just generated x, now expect y
            expecting_x_token[sample_idx] = False
        else:
            # We just generated y, now expect x for next coordinate pair
            expecting_x_token[sample_idx] = True
            
            # We completed a (x,y) pair
            if len(batch_decoder_tokens[sample_idx]) >= 2:
                x_tok = batch_decoder_tokens[sample_idx][-2]
                y_tok = batch_decoder_tokens[sample_idx][-1]
                
                # Decode coordinates
                coords = self.tokenizer.decode_coordinates([x_tok, y_tok])
                if coords:
                    batch_predicted_coords[sample_idx].append(coords[0])
                else:
                    # Fallback to previous coordinate if decoding fails
                    prev = batch_predicted_coords[sample_idx][-1] if batch_predicted_coords[sample_idx] else (0, 0)
                    batch_predicted_coords[sample_idx].append(prev)
                    print(f"Warning: Failed to decode coordinates [{x_tok}, {y_tok}] for sample {sample_idx}")
                
                # Check if we've generated enough coordinates
                if len(batch_predicted_coords[sample_idx]) >= len(batch_future_temporal[sample_idx]):
                    return True  # Finished
        
        return False  # Not finished