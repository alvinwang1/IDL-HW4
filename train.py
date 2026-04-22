import yaml
import gc
import torch
import os
import json
import wandb
import pandas as pd
from torch.utils.data import DataLoader
from hw4lib.data import H4Tokenizer, ASRDataset, verify_dataloader
from hw4lib.model import EncoderDecoderTransformer
from hw4lib.trainers import ASRTrainer
from hw4lib.utils import create_scheduler, create_optimizer

def main():
    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    # Set config data root to Modal's mounted volume
    config['data']['root'] = "/data/hw4p2_data"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize Tokenizer
    Tokenizer = H4Tokenizer(
        token_map  = config['tokenization']['token_map'],
        token_type = config['tokenization']['token_type']
    )
    
    print("Loading datasets...")
    train_dataset = ASRDataset(
        partition=config['data']['train_partition'],
        config=config['data'],
        tokenizer=Tokenizer,
        isTrainPartition=True,
        global_stats=None
    )
    
    global_stats = None
    if config['data']['norm'] == 'global_mvn':
        global_stats = (train_dataset.global_mean, train_dataset.global_std)
        print(f"Global stats computed from training set.")
        
    val_dataset = ASRDataset(
        partition=config['data']['val_partition'],
        config=config['data'],
        tokenizer=Tokenizer,
        isTrainPartition=False,
        global_stats=global_stats
    )
    
    test_dataset = ASRDataset(
        partition=config['data']['test_partition'],
        config=config['data'],
        tokenizer=Tokenizer,
        isTrainPartition=False,
        global_stats=global_stats
    )
    
    gc.collect()

    print("Creating dataloaders...")
    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = config['data']['batch_size'],
        shuffle     = True,
        num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
        pin_memory  = True,
        collate_fn  = train_dataset.collate_fn
    )

    val_loader = DataLoader(
        dataset     = val_dataset,
        batch_size  = config['data']['batch_size'],
        shuffle     = False,
        num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
        pin_memory  = True,
        collate_fn  = val_dataset.collate_fn
    )

    test_loader = DataLoader(
        dataset     = test_dataset,
        batch_size  = config['data']['batch_size'],
        shuffle     = False,
        num_workers = config['data']['NUM_WORKERS'] if device == 'cuda' else 0,
        pin_memory  = True,
        collate_fn  = test_dataset.collate_fn
    )

    max_feat_len       = max(train_dataset.feat_max_len, val_dataset.feat_max_len, test_dataset.feat_max_len)
    max_transcript_len = max(train_dataset.text_max_len, val_dataset.text_max_len, test_dataset.text_max_len)
    max_len            = max(max_feat_len, max_transcript_len)
    
    print("="*50)
    print(f"{'Max Feature Length':<30} : {max_feat_len}")
    print(f"{'Max Transcript Length':<30} : {max_transcript_len}")
    print(f"{'Overall Max Length':<30} : {max_len}")
    print("="*50)

    if config['training'].get('use_wandb', False):
        user_key = os.environ.get("WANDB_API_KEY", None)
        if user_key:
            wandb.login(key=user_key)
            
    # Model Setup
    model_config = config['model'].copy()
    model_config.update({
        'max_len': max_len,
        'num_classes': Tokenizer.vocab_size
    })

    print("Initializing model...")
    model = EncoderDecoderTransformer(**model_config)
    
    # Initialize Trainer
    trainer = ASRTrainer(
        model=model,
        tokenizer=Tokenizer,
        config=config,
        run_name="hw4p2_modal_run",
        config_file="config.yaml",
        device=device
    )
    
    # Init optimizer
    trainer.optimizer = create_optimizer(
        model=model,
        opt_config=config['optimizer']
    )
    
    # Init scheduler
    trainer.scheduler = create_scheduler(
        optimizer=trainer.optimizer,
        scheduler_config=config['scheduler'],
        train_loader=train_loader,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )
    
    # Resume from wandb if requested
    if config['training'].get('resume', False) and config['training'].get('wandb_run_id', 'none') != 'none':
        run_id = config['training']['wandb_run_id']
        print(f"Attempting to resume from wandb run {run_id}...")
        try:
            api = wandb.Api()
            project = config['training'].get('wandb_project', 'hw4p2')
            # Entity name is derived from user's notebook outputs
            run = api.run(f"alvinw2-carnegie-mellon-university/{project}/{run_id}")
            
            for file in run.files():
                if file.name.endswith("checkpoint-last-epoch-model.pth"):
                    print(f"Downloading {file.name} from wandb...")
                    file.download(replace=True, root=".")
                    import shutil
                    shutil.copy(file.name, trainer.checkpoint_dir / "checkpoint-last-epoch-model.pth")
                    trainer.load_checkpoint("checkpoint-last-epoch-model.pth")
                    print("Successfully resumed from checkpoint!")
                    break
        except Exception as e:
            print(f"Failed to resume from wandb: {e}")

    # Run training
    # You can change the number of epochs here
    epochs = 50
    print(f"Starting training for {epochs} epochs...")
    trainer.train(train_loader, val_loader, epochs=epochs)
    
if __name__ == "__main__":
    main()
