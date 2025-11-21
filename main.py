"""
Main entry point for Bangla Cyberbullying Detection fine-tuning
Orchestrates the complete training pipeline with K-fold cross-validation
"""

import torch
from transformers import AutoTokenizer
import warnings
import time
import data
import train
from config import parse_arguments, print_config
from utils import set_seed, print_experiment_header


def main():
    """
    Main function that orchestrates the training pipeline.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Parse configuration
    config = parse_arguments()
    config.freeze_base = True
    # Set random seed for reproducibility (MUST be done before any random operations)
    set_seed(config.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Print configuration
    print_config(config)
    
    start_time = time.time() 

    # Initialize tokenizer (using AutoTokenizer for model flexibility)
    print(f"\nLoading tokenizer: {config.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        print(f"Tokenizer loaded successfully. Vocabulary size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Make sure the model name/path is correct and you have internet connection.")
        return
    
    # Load and preprocess data
    print(f"\nLoading dataset from: {config.dataset_path}")
    try:
        comments, labels = data.load_and_preprocess_data(config.dataset_path)
        print(f"Successfully loaded {len(comments)} samples with {len(data.LABEL_COLUMNS)} labels.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {config.dataset_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Print experiment header
    print_experiment_header(config)
    
    # Check if using multilabel stratification
    if config.stratification_type == 'multilabel':
        try:
            import iterstrat
            print("✓ iterative-stratification package found. Multi-label stratification will be used.")
        except ImportError:
            print("⚠ WARNING: iterative-stratification package not found.")
            print("Install it with: pip install iterative-stratification")
            print("Falling back to regular K-fold splitting.")
            config.stratification_type = 'none'
    
    # Run K-fold cross-validation training
    print("\nStarting K-fold cross-validation training...")
    print("-" * 60)
    
    try:
        train.run_kfold_training(config, comments, labels, tokenizer, device)
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user.")
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU out of memory error!")
        print("Try reducing batch size or sequence length.")
        print(f"Current settings: batch_size={config.batch}, max_length={config.max_length}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()