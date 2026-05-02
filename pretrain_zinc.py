import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.utils.seed import set_seed
from src.data.featurizer import MolFeaturizer
from src.models.hybrid_model import GINMambaHybrid
from src.ordering.atomic_number import get_order as atomic_number_order


class SmilesFeaturizer(torch.utils.data.Dataset):
    def __init__(self, smiles_list, featurizer):
        self.smiles_list = smiles_list
        self.featurizer = featurizer

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        try:
            # Use the exact same processing pipeline as the main model
            # Create a PyG Data object with SMILES, just like Tox21Dataset does
            from torch_geometric.data import Data
            data = Data(smiles=smiles)
            
            # Apply the same featurizer used in main training
            data = self.featurizer(data)
            
            # Extract electrochemical properties from the node features
            # The MolFeaturizer already computes and includes these properties
            # Gasteiger charges, LogP, and MR are at indices 40, 41, 42 respectively
            # This leverages the existing featurization without duplication
            
            # Feature indices based on get_node_features:
            # - Atomic number: 11 features
            # - Degree: 6 features  
            # - Hybridization: 6 features
            # - Formal charge: 5 features
            # - Num H: 5 features
            # - Aromatic: 1 feature
            # - In ring: 1 feature
            # - Chirality: 4 features
            # - Radical: 1 feature
            # - Electronic (Gasteiger, LogP, MR): 3 features at indices 40, 41, 42
            
            if data.x.size(1) > 42:  # Make sure we have electronic properties
                gasteiger_values = data.x[:, 40]  # Gasteiger charge
                logp_values = data.x[:, 41]       # LogP contribution
                mr_values = data.x[:, 42]         # MR contribution
                
                data.gasteiger = gasteiger_values
                data.logp = logp_values
                data.mr = mr_values
            else:
                # Fallback if features are missing
                data.gasteiger = torch.zeros(data.x.size(0), dtype=torch.float)
                data.logp = torch.zeros(data.x.size(0), dtype=torch.float)
                data.mr = torch.zeros(data.x.size(0), dtype=torch.float)
            
            return data
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return self._create_empty_data()
    
    def _create_empty_data(self):
        """Create empty data object for failed SMILES parsing"""
        from torch_geometric.data import Data
        return Data(x=torch.zeros((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long))


class PretrainingModel(nn.Module):
    def __init__(self, node_features, d_model, gin_hidden=64, gin_layers=3,
                 mamba_state=16, mamba_conv=4, mamba_expand=2, mamba_layers=1,
                 bidirectional=True, dropout=0.0):
        super().__init__()
        self.hybrid = GINMambaHybrid(
            node_features=node_features,
            d_model=d_model,
            gin_hidden=gin_hidden,
            gin_layers=gin_layers,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            mamba_layers=mamba_layers,
            bidirectional=bidirectional,
            mlp_hidden=64,  # Match main model defaults
            mlp_layers=2,   # Match main model defaults
            num_tasks=1,    # Single task for pre-training
            dropout=dropout,
        )
        # Reconstruction head for node features (MAM)
        self.recon_head = nn.Linear(d_model, node_features)
        
        # ESF heads for electrochemical properties
        self.gasteiger_head = nn.Linear(d_model, 1)
        self.logp_head = nn.Linear(d_model, 1) 
        self.mr_head = nn.Linear(d_model, 1)
        
        # Mask token: A learnable parameter to replace masked node features
        self.mask_token = nn.Parameter(torch.zeros(1, node_features))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, data, mask_ratio=0.15):
        x_orig = data.x.float()
        
        # Generate mask for MAM
        mask = torch.rand(x_orig.size(0), device=x_orig.device) < mask_ratio
        
        # Apply mask
        x_masked = x_orig.clone()
        if mask.any():
            x_masked[mask] = self.mask_token.expand(mask.sum(), -1)

        # Update data object for the hybrid encoder
        data_masked = data.clone()
        data_masked.x = x_masked

        # Forward through GIN-Mamba
        h_fused = self.hybrid.encode_atoms(data_masked, atomic_number_order)

        # MAM Reconstruction Loss
        recon = self.recon_head(h_fused)
        mam_loss = F.mse_loss(recon[mask], x_orig[mask], reduction='mean') if mask.any() else torch.tensor(0.0, device=x_orig.device, requires_grad=True)

        # ESF Electrochemical Prediction Loss
        # Extract target electrochemical properties from data
        if hasattr(data, 'gasteiger') and hasattr(data, 'logp') and hasattr(data, 'mr'):
            gasteiger_pred = self.gasteiger_head(h_fused).squeeze(-1)
            logp_pred = self.logp_head(h_fused).squeeze(-1)
            mr_pred = self.mr_head(h_fused).squeeze(-1)
            
            # Check if electrochemical properties have valid values
            gasteiger_valid = data.gasteiger.float().abs().sum() > 0
            logp_valid = data.logp.float().abs().sum() > 0
            mr_valid = data.mr.float().abs().sum() > 0
            
            if gasteiger_valid:
                gasteiger_loss = F.mse_loss(gasteiger_pred, data.gasteiger.float(), reduction='mean')
            else:
                gasteiger_loss = torch.tensor(0.0, device=x_orig.device, requires_grad=True)
            
            if logp_valid:
                logp_loss = F.mse_loss(logp_pred, data.logp.float(), reduction='mean')
            else:
                logp_loss = torch.tensor(0.0, device=x_orig.device, requires_grad=True)
            
            if mr_valid:
                mr_loss = F.mse_loss(mr_pred, data.mr.float(), reduction='mean')
            else:
                mr_loss = torch.tensor(0.0, device=x_orig.device, requires_grad=True)
            
            esf_loss = (gasteiger_loss + logp_loss + mr_loss) / 3.0
        else:
            esf_loss = torch.tensor(0.0, device=x_orig.device, requires_grad=True)

        return mam_loss, esf_loss, mask.sum()


def setup_logger(log_dir="outputs/logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"pretrain_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def pretrain_epoch(model, dataloader, optimizer, device, esf_weight=0.1, mask_ratio=0.15):
    model.train()
    total_mam = 0.0
    total_esf = 0.0
    total_masked = 0.0
    num_batches = len(dataloader)
    use_amp = device.type == "cuda"

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            mam_loss, esf_loss, n_masked = model(batch, mask_ratio=mask_ratio)
            loss = mam_loss + esf_weight * esf_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_mam += mam_loss.item()
        total_esf += esf_loss.item()
        total_masked += n_masked.item()

    avg_mam = total_mam / max(num_batches, 1)
    avg_esf = total_esf / max(num_batches, 1)
    avg_masked = total_masked / max(num_batches, 1)
    return avg_mam, avg_esf, avg_masked


def validate_epoch(model, dataloader, device, mask_ratio=0.15):
    model.eval()
    total_mam = 0.0
    total_esf = 0.0
    total_masked = 0.0
    
    # ESF component losses
    total_gasteiger_loss = 0.0
    total_logp_loss = 0.0
    total_mr_loss = 0.0
    
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Get predictions for detailed ESF analysis
            x_orig = batch.x.float()
            mask = torch.rand(x_orig.size(0), device=x_orig.device) < mask_ratio
            
            x_masked = x_orig.clone()
            if mask.any():
                x_masked[mask] = model.mask_token.expand(mask.sum(), -1)
            
            data_masked = batch.clone()
            data_masked.x = x_masked
            
            h_fused = model.hybrid.encode_atoms(data_masked, atomic_number_order)
            
            # MAM reconstruction
            recon = model.recon_head(h_fused)
            mam_loss = F.mse_loss(recon[mask], x_orig[mask], reduction='mean') if mask.any() else torch.tensor(0.0, device=x_orig.device)
            
            # ESF predictions
            if hasattr(batch, 'gasteiger') and hasattr(batch, 'logp') and hasattr(batch, 'mr'):
                gasteiger_pred = model.gasteiger_head(h_fused).squeeze(-1)
                logp_pred = model.logp_head(h_fused).squeeze(-1)
                mr_pred = model.mr_head(h_fused).squeeze(-1)
                
                # Check if electrochemical properties have valid values
                gasteiger_valid = batch.gasteiger.float().abs().sum() > 0
                logp_valid = batch.logp.float().abs().sum() > 0
                mr_valid = batch.mr.float().abs().sum() > 0
                
                if gasteiger_valid:
                    gasteiger_loss = F.mse_loss(gasteiger_pred, batch.gasteiger.float(), reduction='mean')
                    total_gasteiger_loss += gasteiger_loss.item()
                else:
                    gasteiger_loss = torch.tensor(0.0, device=x_orig.device)
                
                if logp_valid:
                    logp_loss = F.mse_loss(logp_pred, batch.logp.float(), reduction='mean')
                    total_logp_loss += logp_loss.item()
                else:
                    logp_loss = torch.tensor(0.0, device=x_orig.device)
                
                if mr_valid:
                    mr_loss = F.mse_loss(mr_pred, batch.mr.float(), reduction='mean')
                    total_mr_loss += mr_loss.item()
                else:
                    mr_loss = torch.tensor(0.0, device=x_orig.device)
                
                esf_loss = (gasteiger_loss + logp_loss + mr_loss) / 3.0
            else:
                esf_loss = torch.tensor(0.0, device=x_orig.device)
            
            total_mam += mam_loss.item()
            total_esf += esf_loss.item()
            total_masked += mask.sum().item() if mask.any() else 0

    avg_mam = total_mam / max(num_batches, 1)
    avg_esf = total_esf / max(num_batches, 1)
    avg_masked = total_masked / max(num_batches, 1)
    
    # Detailed ESF metrics
    esf_metrics = {
        'mam_loss': avg_mam,
        'esf_loss': avg_esf,
        'gasteiger_loss': total_gasteiger_loss / max(num_batches, 1) if total_gasteiger_loss > 0 else 0.0,
        'logp_loss': total_logp_loss / max(num_batches, 1) if total_logp_loss > 0 else 0.0,
        'mr_loss': total_mr_loss / max(num_batches, 1) if total_mr_loss > 0 else 0.0,
        'avg_masked': avg_masked
    }
    
    return esf_metrics


def main():
    parser = argparse.ArgumentParser(description="Pretrain GIN-Mamba Hybrid on ZINC")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to config file")
    parser.add_argument("--zinc_root", type=str, default="data/ZINC", help="ZINC dataset root path")
    parser.add_argument("--subset", action="store_true", default=True, help="Use ZINC-12K subset")
    parser.add_argument("--max_molecules", type=int, default=None, help="Limit molecules (None = use all)")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    logger = setup_logger()
    logger.info(f"Arguments: {args}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    rwse_walk_length = config.get("model", {}).get("rwse_walk_length", 16)
    featurizer = MolFeaturizer(rwse_walk_length=rwse_walk_length)

    logger.info("Loading ZINC dataset...")
    import pandas as pd
    zinc_csv_path = os.path.join(args.zinc_root, 'raw', 'zinc250k.csv')

    
    zinc_csv = pd.read_csv(zinc_csv_path)
    zinc_smiles = [smiles for smiles in zinc_csv['smiles'] if isinstance(smiles, str)]
    logger.info(f"Loaded {len(zinc_smiles)} molecules from zinc250k.csv")

    max_molecules = config.get("data", {}).get("max_molecules", 50000)
    zinc_smiles = zinc_smiles[:min(max_molecules, len(zinc_smiles))]
    logger.info(f"Using {len(zinc_smiles)} molecules for pre-training")

    dataset = SmilesFeaturizer(zinc_smiles, featurizer)
    
    # Filter out invalid data
    valid_indices = []
    for i in range(len(dataset)):
        data = dataset[i]
        if data.x.size(0) > 0 and data.edge_index.size(1) > 0:
            valid_indices.append(i)
    
    # Create subset with valid molecules
    dataset = torch.utils.data.Subset(dataset, valid_indices)
    logger.info(f"Filtered to {len(dataset)} valid molecules")
    
    sample_data = dataset[0]
    node_features = sample_data.x.size(1)
    logger.info(f"Node feature dimension: {node_features}")

    batch_size = args.batch_size or config["data"]["batch_size"]

    m_cfg = config["model"]
    model = PretrainingModel(
        node_features=node_features,
        d_model=m_cfg["d_model"],
        gin_hidden=m_cfg.get("gin_hidden", 64),
        gin_layers=m_cfg["gin_layers"],
        mamba_state=m_cfg["mamba_state"],
        mamba_conv=m_cfg["mamba_conv"],
        mamba_expand=m_cfg.get("mamba_expand", 2),
        mamba_layers=m_cfg["mamba_layers"],
        bidirectional=m_cfg.get("bidirectional", True),
        dropout=m_cfg.get("dropout", 0.0),
    ).to(device)

    lr = args.lr or float(config["training"]["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
        fused=device.type == "cuda",
    )

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    esf_weight = config["training"].get("esf_weight", 0.1)
    mask_ratio = config["training"].get("mask_ratio", 0.15)

    # Create validation dataloader
    val_size = min(1000, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    logger.info(f"Starting pretraining for {epochs} epochs...")
    
    best_loss = float("inf")
    best_val_loss = float("inf")
    checkpoint_path = "outputs/checkpoints/pretrained_best.pt"
    patience = 10
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training epoch
        train_mam_loss, train_esf_loss, train_avg_masked = pretrain_epoch(
            model, train_loader, optimizer, device,
            esf_weight=esf_weight, mask_ratio=mask_ratio,
        )
        train_total_loss = train_mam_loss + esf_weight * train_esf_loss
        
        # Validation epoch
        val_metrics = validate_epoch(model, val_loader, device, mask_ratio=mask_ratio)
        val_total_loss = val_metrics['mam_loss'] + esf_weight * val_metrics['esf_loss']
        
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train MAM: {train_mam_loss:.6f} | Train ESF: {train_esf_loss:.6f} | Train Total: {train_total_loss:.6f} | "
            f"Val MAM: {val_metrics['mam_loss']:.6f} | Val ESF: {val_metrics['esf_loss']:.6f} | Val Total: {val_total_loss:.6f} | "
            f"Gasteiger: {val_metrics['gasteiger_loss']:.6f} | LogP: {val_metrics['logp_loss']:.6f} | MR: {val_metrics['mr_loss']:.6f} | "
            f"Masked: {train_avg_masked:.1f}/{val_metrics['avg_masked']:.1f}"
        )

        # Early stopping and checkpointing
        improvement = val_total_loss < best_val_loss - 0.0001
        if improvement:
            best_val_loss = val_total_loss
            best_loss = train_total_loss
            
            # Create state dict with proper prefix for compatibility with main model
            pretrained_state_dict = {}
            for key, value in model.state_dict().items():
                if key.startswith("hybrid."):
                    # Keep hybrid model keys as-is
                    pretrained_state_dict[key] = value
                # Skip pretraining-only components (recon_head, esf_head, mask_token)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Full model for resuming
                'pretrained_state_dict': pretrained_state_dict,  # For fine-tuning
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_total_loss,
                'val_loss': val_total_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"-> Saved best pretrained model (val_loss={best_val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break

    # Load best model for final save
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create state dict with proper prefix for compatibility with main model
    pretrained_state_dict = {}
    for key, value in model.state_dict().items():
        if key.startswith("hybrid."):
            # Keep hybrid model keys as-is
            pretrained_state_dict[key] = value
        else:
            # Skip pretraining-only components (recon_head, esf_head, mask_token)
            continue
    
    final_path = f"outputs/checkpoints/pretrained_final_epoch{epoch}.pt"
    torch.save(pretrained_state_dict, final_path)
    logger.info(f"Saved final pretrained model (hybrid weights only) to {final_path}")
    logger.info("Pretraining completed.")


if __name__ == "__main__":
    main()
