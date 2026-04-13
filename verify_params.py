import torch
from hw4lib.model import EncoderDecoderTransformer
from torchinfo import summary

def verify():
    # Model parameters based on user's torchinfo summary
    config = {
        'input_dim': 80,
        'time_reduction': 4,
        'reduction_method': 'both',
        'num_encoder_layers': 6,
        'num_encoder_heads': 8,
        'd_ff_encoder': 980,
        'num_decoder_layers': 2,
        'num_decoder_heads': 8,
        'd_ff_decoder': 980,
        'd_model': 512,
        'dropout': 0.1,
        'max_len': 3000,
        'num_classes': 5000,
        'weight_tying': True
    }
    
    # Instantiate model
    model = EncoderDecoderTransformer(**config)
    
    # Dummy inputs
    padded_feats = torch.randn(2, 1000, 80)
    padded_shifted = torch.randint(0, 5000, (2, 500))
    feat_lengths = torch.tensor([1000, 800])
    transcript_lengths = torch.tensor([500, 400])
    
    # Verification
    print("\n" + "="*50)
    print("VERIFYING PARAMETER COUNT")
    print("="*50)
    
    model_stats = summary(model, input_data=[padded_feats, padded_shifted, feat_lengths, transcript_lengths])
    print(model_stats)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFinal Trainable Parameters: {total_params:,}")
    
    if total_params < 30_000_000:
        print("\n✅ SUCCESS: Model is under 30M limit!")
    else:
        print(f"\n❌ FAILURE: Model is at {total_params:,} (Over limit by {total_params - 30_000_000:,})")

if __name__ == "__main__":
    verify()
