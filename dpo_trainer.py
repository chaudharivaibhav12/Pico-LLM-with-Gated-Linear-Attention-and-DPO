import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import argparse
from typing import List, Tuple
import wandb

# Import your existing model classes
# Assuming these are available from pico-llm.py
# from pico_llm import TransformerModel, seq_collate_fn
from pico_llm_kimi_claude import KimiLinearTransformer, seq_collate_fn
import tiktoken


class DPOPreferenceDataset(Dataset):
    """
    Dataset for DPO training containing preference pairs.
    
    Each example contains:
    - prompt: the input prompt
    - chosen: the preferred completion
    - rejected: the less preferred completion
    """
    
    def __init__(self, prompts: List[str], chosen: List[str], rejected: List[str], 
                tokenizer, max_length: int = 512):
        """
        Args:
            prompts: List of prompt strings
            chosen: List of preferred completion strings
            rejected: List of rejected completion strings
            tokenizer: Tiktoken tokenizer
            max_length: Maximum sequence length
        """
        assert len(prompts) == len(chosen) == len(rejected), \
            "Prompts, chosen, and rejected must have same length"
        
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        """
        Returns:
            prompt_ids: tokenized prompt
            chosen_ids: tokenized chosen completion (prompt + chosen)
            rejected_ids: tokenized rejected completion (prompt + rejected)
        """
        prompt = self.prompts[idx]
        chosen_text = prompt + self.chosen[idx]
        rejected_text = prompt + self.rejected[idx]
        
        prompt_ids = self.tokenizer.encode(prompt)[:self.max_length]
        chosen_ids = self.tokenizer.encode(chosen_text)[:self.max_length]
        rejected_ids = self.tokenizer.encode(rejected_text)[:self.max_length]
        
        return {
            'prompt_ids': torch.tensor(prompt_ids, dtype=torch.long),
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'prompt_length': len(prompt_ids)
        }


def dpo_collate_fn(batch):
    """
    Collate function for DPO dataset.
    Pads sequences to the max length in the batch.
    
    Returns:
        Dictionary with keys:
        - prompt_ids: (max_prompt_len, batch_size)
        - chosen_ids: (max_chosen_len, batch_size)
        - rejected_ids: (max_rejected_len, batch_size)
        - prompt_lengths: (batch_size,)
    """
    prompt_ids = [item['prompt_ids'] for item in batch]
    chosen_ids = [item['chosen_ids'] for item in batch]
    rejected_ids = [item['rejected_ids'] for item in batch]
    prompt_lengths = torch.tensor([item['prompt_length'] for item in batch])
    
    # Find max lengths
    max_prompt_len = max(len(p) for p in prompt_ids)
    max_chosen_len = max(len(c) for c in chosen_ids)
    max_rejected_len = max(len(r) for r in rejected_ids)
    
    batch_size = len(batch)
    
    # Pad and stack
    prompt_padded = torch.zeros(max_prompt_len, batch_size, dtype=torch.long)
    chosen_padded = torch.zeros(max_chosen_len, batch_size, dtype=torch.long)
    rejected_padded = torch.zeros(max_rejected_len, batch_size, dtype=torch.long)
    
    for i in range(batch_size):
        prompt_padded[:len(prompt_ids[i]), i] = prompt_ids[i]
        chosen_padded[:len(chosen_ids[i]), i] = chosen_ids[i]
        rejected_padded[:len(rejected_ids[i]), i] = rejected_ids[i]
    
    return {
        'prompt_ids': prompt_padded,
        'chosen_ids': chosen_padded,
        'rejected_ids': rejected_padded,
        'prompt_lengths': prompt_lengths
    }


def compute_sequence_log_probs(model, tokens, prompt_length):
    """
    Compute log probabilities for a sequence under the model.
    
    Args:
        model: TransformerModel
        tokens: (seq_len, batch_size) token IDs
        prompt_length: length of the prompt (don't compute loss on prompt)
    
    Returns:
        log_probs: (batch_size,) sum of log probs for completion tokens
    """
    seq_len, batch_size = tokens.shape
    
    # Forward pass
    logits, _ = model(tokens)  # (seq_len, batch_size, vocab_size)
    
    # Compute log probabilities
    log_probs_all = F.log_softmax(logits, dim=-1)  # (seq_len, batch_size, vocab_size)
    
    # Get log prob of actual next tokens (shift by 1)
    # For each position t, we want log_prob of token[t+1]
    if seq_len < 2:
        return torch.zeros(batch_size, device=tokens.device)
    
    preds = log_probs_all[:-1, :, :]  # (seq_len-1, batch_size, vocab_size)
    targets = tokens[1:, :]  # (seq_len-1, batch_size)
    
    # Gather the log probs of the target tokens
    target_log_probs = torch.gather(
        preds, 
        dim=2, 
        index=targets.unsqueeze(2)
    ).squeeze(2)  # (seq_len-1, batch_size)
    
    # Only sum log probs for completion (not prompt)
    # Create mask: 1 for completion tokens, 0 for prompt tokens
    mask = torch.arange(seq_len - 1, device=tokens.device).unsqueeze(1) >= prompt_length
    mask = mask.float()
    
    # Sum log probs for completion tokens
    masked_log_probs = target_log_probs * mask
    sequence_log_probs = masked_log_probs.sum(dim=0)  # (batch_size,)
    
    return sequence_log_probs


def dpo_loss(policy_chosen_log_probs, policy_rejected_log_probs,
            ref_chosen_log_probs, ref_rejected_log_probs, beta=0.1):
    """
    Compute DPO loss.
    
    The DPO loss is:
    L = -log(sigmoid(beta * (log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected))))
    
    Where:
    - pi is the policy model being trained
    - pi_ref is the frozen reference model
    - beta is a temperature parameter (typically 0.1)
    
    Args:
        policy_chosen_log_probs: log probs of chosen completions under policy
        policy_rejected_log_probs: log probs of rejected completions under policy
        ref_chosen_log_probs: log probs of chosen completions under reference
        ref_rejected_log_probs: log probs of rejected completions under reference
        beta: temperature parameter
    
    Returns:
        loss: scalar DPO loss
        metrics: dictionary of useful metrics
    """
    # Compute log ratios
    policy_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
    ref_log_ratios = ref_chosen_log_probs - ref_rejected_log_probs
    
    # DPO objective
    logits = beta * (policy_log_ratios - ref_log_ratios)
    loss = -F.logsigmoid(logits).mean()
    
    # Compute implicit reward (for monitoring)
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_log_probs - ref_chosen_log_probs)
        rejected_rewards = beta * (policy_rejected_log_probs - ref_rejected_log_probs)
        reward_margin = chosen_rewards - rejected_rewards
        accuracy = (policy_log_ratios > ref_log_ratios).float().mean()
    
    metrics = {
        'loss': loss.item(),
        'chosen_rewards': chosen_rewards.mean().item(),
        'rejected_rewards': rejected_rewards.mean().item(),
        'reward_margin': reward_margin.mean().item(),
        'accuracy': accuracy.item()
    }
    
    return loss, metrics


class DPOTrainer:
    """
    Trainer for DPO (Direct Preference Optimization).
    """
    
    def __init__(self, policy_model, ref_model, tokenizer, device='cuda', 
                beta=0.1, learning_rate=1e-6):
        """
        Args:
            policy_model: The model to train (your TransformerModel)
            ref_model: Frozen reference model (copy of initial policy)
            tokenizer: Tiktoken tokenizer
            device: torch device
            beta: DPO temperature parameter
            learning_rate: learning rate for optimizer
        """
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()  # Freeze reference model
        
        # Freeze reference model parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=learning_rate)
        
    def train_step(self, batch):
        """
        Perform one training step.
        
        Args:
            batch: Dictionary from dpo_collate_fn
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.policy_model.train()
        
        # Move batch to device
        prompt_ids = batch['prompt_ids'].to(self.device)
        chosen_ids = batch['chosen_ids'].to(self.device)
        rejected_ids = batch['rejected_ids'].to(self.device)
        prompt_lengths = batch['prompt_lengths'].to(self.device)
        
        # Compute log probs under policy model
        policy_chosen_log_probs = compute_sequence_log_probs(
            self.policy_model, chosen_ids, prompt_lengths[0]
        )
        policy_rejected_log_probs = compute_sequence_log_probs(
            self.policy_model, rejected_ids, prompt_lengths[0]
        )
        
        # Compute log probs under reference model (no grad)
        with torch.no_grad():
            ref_chosen_log_probs = compute_sequence_log_probs(
                self.ref_model, chosen_ids, prompt_lengths[0]
            )
            ref_rejected_log_probs = compute_sequence_log_probs(
                self.ref_model, rejected_ids, prompt_lengths[0]
            )
        
        # Compute DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_log_probs, policy_rejected_log_probs,
            ref_chosen_log_probs, ref_rejected_log_probs,
            beta=self.beta
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        return metrics
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test set.
        
        Args:
            test_loader: DataLoader with test preference data
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.policy_model.eval()
        
        total_metrics = {
            'loss': 0.0,
            'chosen_rewards': 0.0,
            'rejected_rewards': 0.0,
            'reward_margin': 0.0,
            'accuracy': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                prompt_ids = batch['prompt_ids'].to(self.device)
                chosen_ids = batch['chosen_ids'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)
                prompt_lengths = batch['prompt_lengths'].to(self.device)
                
                # Compute log probs under policy model
                policy_chosen_log_probs = compute_sequence_log_probs(
                    self.policy_model, chosen_ids, prompt_lengths[0]
                )
                policy_rejected_log_probs = compute_sequence_log_probs(
                    self.policy_model, rejected_ids, prompt_lengths[0]
                )
                
                # Compute log probs under reference model
                ref_chosen_log_probs = compute_sequence_log_probs(
                    self.ref_model, chosen_ids, prompt_lengths[0]
                )
                ref_rejected_log_probs = compute_sequence_log_probs(
                    self.ref_model, rejected_ids, prompt_lengths[0]
                )
                
                # Compute DPO loss
                loss, metrics = dpo_loss(
                    policy_chosen_log_probs, policy_rejected_log_probs,
                    ref_chosen_log_probs, ref_rejected_log_probs,
                    beta=self.beta
                )
                
                # Accumulate metrics
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        self.policy_model.train()
        return total_metrics
    
    def train(self, train_loader, test_loader, num_epochs, log_interval=10):
        """
        Train the model with DPO.
        
        Args:
            train_loader: DataLoader with training preference data
            test_loader: DataLoader with test preference data
            num_epochs: number of training epochs
            log_interval: log metrics every N steps
        """
        wandb_run = wandb.init(project="pico-llm-dpo", config={
            'beta': self.beta,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'num_epochs': num_epochs,
            'train_size': len(train_loader.dataset),
            'test_size': len(test_loader.dataset)
        })
        
        global_step = 0
        best_test_accuracy = 0.0
        
        for epoch in range(1, num_epochs + 1):
            # ========== TRAINING ==========
            epoch_metrics = {
                'loss': 0.0,
                'chosen_rewards': 0.0,
                'rejected_rewards': 0.0,
                'reward_margin': 0.0,
                'accuracy': 0.0
            }
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader, 1):
                metrics = self.train_step(batch)
                global_step += 1
                
                # Accumulate metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
                
                # Log to wandb
                if batch_idx % log_interval == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Step {batch_idx}/{len(train_loader)}")
                    print(f"  Train Loss: {metrics['loss']:.4f}")
                    print(f"  Train Reward Margin: {metrics['reward_margin']:.4f}")
                    print(f"  Train Accuracy: {metrics['accuracy']:.4f}")
                    
                    wandb_run.log({
                        'train/step_loss': metrics['loss'],
                        'train/step_chosen_rewards': metrics['chosen_rewards'],
                        'train/step_rejected_rewards': metrics['rejected_rewards'],
                        'train/step_reward_margin': metrics['reward_margin'],
                        'train/step_accuracy': metrics['accuracy'],
                        'epoch': epoch
                    }, step=global_step)
            
            # Average training metrics for epoch
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            # ========== EVALUATION ==========
            print(f"\nEvaluating on test set...")
            test_metrics = self.evaluate(test_loader)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {epoch_metrics['loss']:.4f} | Test Loss: {test_metrics['loss']:.4f}")
            print(f"  Train Reward Margin: {epoch_metrics['reward_margin']:.4f} | Test Reward Margin: {test_metrics['reward_margin']:.4f}")
            print(f"  Train Accuracy: {epoch_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")
            
            # Check for best model
            if test_metrics['accuracy'] > best_test_accuracy:
                best_test_accuracy = test_metrics['accuracy']
                print(f"  ✓ New best test accuracy: {best_test_accuracy:.4f}")
            
            # Log epoch metrics to wandb
            wandb_run.log({
                'train/epoch_loss': epoch_metrics['loss'],
                'train/epoch_reward_margin': epoch_metrics['reward_margin'],
                'train/epoch_accuracy': epoch_metrics['accuracy'],
                'test/epoch_loss': test_metrics['loss'],
                'test/epoch_reward_margin': test_metrics['reward_margin'],
                'test/epoch_accuracy': test_metrics['accuracy'],
                'test/best_accuracy': best_test_accuracy,
                'epoch': epoch
            }, step=global_step)
            
            print()
        
        print(f"\nTraining complete!")
        print(f"Best test accuracy: {best_test_accuracy:.4f}")
        
        wandb_run.log({
            'final/best_test_accuracy': best_test_accuracy
        })
        
        wandb_run.finish()
        print("DPO training complete!")


def load_pretrained_model(checkpoint_path, vocab_size, embed_size, device, block_size, n_heads, n_blocks):
    """
    Load a pretrained TransformerModel from checkpoint.
    
    Args:
        checkpoint_path: path to .pt file
        vocab_size: vocabulary size
        embed_size: model dimension
        device: torch device
    
    Returns:
        model: loaded TransformerModel
        block_size: the model's block_size
    """
    model = KimiLinearTransformer(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=n_heads,
        n_blocks=n_blocks,
        block_size=block_size,
        use_pre_norm=True
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Get the actual block_size from the loaded model
    block_size = model.block_size

    print(f"Loaded model from {checkpoint_path}")
    print(f"Model block_size: {block_size}")
    return model, block_size


def load_preference_data_from_hf(dataset_name="HuggingFaceH4/ultrafeedback_binarized",
    split="train_prefs",
    max_samples=None):
    """
    Load preference data from HuggingFace datasets.
    
    Supports multiple popular DPO datasets:
    - Anthropic/hh-rlhf (helpful-base, helpful-online, harmless-base, etc.)
    - HuggingFaceH4/ultrafeedback_binarized
    - lvwerra/stack-exchange-paired
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: "train" or "test"
        max_samples: Limit number of samples (None for all)
    
    Returns:
        prompts, chosen, rejected lists
    """
    from datasets import load_dataset
    from tqdm import tqdm
    
    print(f"Loading {dataset_name} {split} split)...")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except:
        raise ValueError(f"Could not load dataset {dataset_name}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    prompts = []
    chosen = []
    rejected = []
    
    for example in tqdm(dataset, desc="Processing"):
        prompt_text = example['prompt']
        chosen_msgs = example['chosen']
        rejected_msgs = example['rejected']
        
        if chosen_msgs and rejected_msgs:
            chosen_response = chosen_msgs[-1]['content']
            rejected_response = rejected_msgs[-1]['content']
            
            prompts.append(prompt_text)
            chosen.append(chosen_response)
            rejected.append(rejected_response)
    
    print(f"Loaded {len(prompts)} preference pairs")
    return prompts, chosen, rejected


def load_preference_data_from_jsonl(filepath):
    """
    Load preference data from a JSONL file.
    
    Expected format per line:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    import json
    
    prompts, chosen, rejected = [], [], []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            prompts.append(example['prompt'])
            chosen.append(example['chosen'])
            rejected.append(example['rejected'])
    
    print(f"Loaded {len(prompts)} examples from {filepath}")
    return prompts, chosen, rejected


def main():
    parser = argparse.ArgumentParser(description="DPO training for TransformerModel")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained model checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for DPO training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of DPO training epochs")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter (default: 0.1)")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate (default: 1e-6)")
    
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Block size for DPO training")
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Model embedding dimension")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=6,  # Set default to 6
        help="Number of Transformer blocks/layers. Default=6.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,  # Set default to 8
        help="Number of attention heads for transformer. Default=8.",
    )
    parser.add_argument("--save_path", type=str, default="dpo_model.pt",
                        help="Path to save DPO-trained model")
    
    # Data loading options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--hf_dataset", type=str,
                        help="HuggingFace dataset name (e.g., 'Anthropic/hh-rlhf')")
    data_group.add_argument("--jsonl_file", type=str,
                           help="Path to JSONL file with preference data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (None for all)")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Fraction of data to use for testing (default: 0.1)")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    # Load pretrained model
    print("Loading pretrained model...")
    policy_model, block_size = load_pretrained_model(
        args.checkpoint, vocab_size, args.embed_size, device, args.block_size, args.n_heads, args.n_blocks
    )
    
    # Create reference model (frozen copy)
    print("Creating reference model...")
    ref_model = copy.deepcopy(policy_model)
    
    # Load preference data
    print("\nLoading preference data...")
    if args.hf_dataset:
        prompts, chosen, rejected = load_preference_data_from_hf(
            dataset_name=args.hf_dataset,
            split="train_prefs",
            max_samples=args.max_samples
        )
    else:
        prompts, chosen, rejected = load_preference_data_from_jsonl(args.jsonl_file)
        if args.max_samples:
            prompts = prompts[:args.max_samples]
            chosen = chosen[:args.max_samples]
            rejected = rejected[:args.max_samples]
    
    # Train-test split
    total_size = len(prompts)
    test_size = int(total_size * args.test_split)
    train_size = total_size - test_size
    
    print(f"\nDataset split:")
    print(f"  Total examples: {total_size}")
    print(f"  Train: {train_size} ({(1-args.test_split)*100:.1f}%)")
    print(f"  Test: {test_size} ({args.test_split*100:.1f}%)")
    
    # Shuffle and split
    import random
    indices = list(range(total_size))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_prompts = [prompts[i] for i in train_indices]
    train_chosen = [chosen[i] for i in train_indices]
    train_rejected = [rejected[i] for i in train_indices]
    
    test_prompts = [prompts[i] for i in test_indices]
    test_chosen = [chosen[i] for i in test_indices]
    test_rejected = [rejected[i] for i in test_indices]
    
    # Show data statistics
    print(f"\nTraining Set Statistics:")
    print(f"  Avg prompt length: {sum(len(p.split()) for p in train_prompts) / len(train_prompts):.1f} words")
    print(f"  Avg chosen length: {sum(len(c.split()) for c in train_chosen) / len(train_chosen):.1f} words")
    print(f"  Avg rejected length: {sum(len(r.split()) for r in train_rejected) / len(train_rejected):.1f} words")
    
    print(f"\nTest Set Statistics:")
    print(f"  Avg prompt length: {sum(len(p.split()) for p in test_prompts) / len(test_prompts):.1f} words")
    print(f"  Avg chosen length: {sum(len(c.split()) for c in test_chosen) / len(test_chosen):.1f} words")
    print(f"  Avg rejected length: {sum(len(r.split()) for r in test_rejected) / len(test_rejected):.1f} words")
    
    # Show example
    print(f"\nExample from training set:")
    print(f"Prompt: {train_prompts[0][:150]}...")
    print(f"Chosen: {train_chosen[0][:150]}...")
    print(f"Rejected: {train_rejected[0][:150]}...")

    # Create datasets and loaders using model's block_size
    print(f"\nUsing max_length={block_size} (model's block_size)")
    train_dataset = DPOPreferenceDataset(train_prompts, train_chosen, train_rejected, enc, max_length=block_size)
    test_dataset = DPOPreferenceDataset(test_prompts, test_chosen, test_rejected, enc, max_length=block_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=dpo_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dpo_collate_fn
    )
    
    print(f"\nTraining configuration:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Beta: {args.beta}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    
    # Create trainer and train
    print("\nStarting DPO training...")
    trainer = DPOTrainer(
        policy_model, 
        ref_model, 
        enc, 
        device=device,
        beta=args.beta,
        learning_rate=args.learning_rate
    )
    
    trainer.train(train_loader, test_loader, num_epochs=args.num_epochs, log_interval=10)
    
    # Save trained model
    print(f"\nSaving DPO-trained model to {args.save_path}...")
    torch.save({
        'model_state_dict': policy_model.state_dict(),
        'beta': args.beta,
        'learning_rate': args.learning_rate,
        'dataset': args.hf_dataset if args.hf_dataset else args.jsonl_file,
        'train_samples': train_size,
        'test_samples': test_size
    }, args.save_path)
    
    print("Done!")


if __name__ == "__main__":
    main()