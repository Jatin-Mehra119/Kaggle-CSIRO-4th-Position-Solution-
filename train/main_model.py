import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class CFG:
    # Paths
    train_csv = 'cleaned-data/train.csv'  
    train_dir = 'cleaned-data/train'      
    output_dir = 'Models_Seeded_Ensemble'
    weights_path = 'weights/vit-huge-plus-patch-16-dino-v3/pytorch_model.bin'


    # Model settings
    model_name = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    img_size = 800
    n_folds = 5
    selected_folds = [0, 1, 2, 3, 4] 

    # Training settings
    epochs = 20
    batch_size = 10
    learning_rate = 5e-5
    weight_decay = 1e-5
    num_workers = 16

    # SEED LIST
    seeds = [42] 

    # Augmentation
    use_augmentation = True

    # Target names (order matters!)
    targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory
os.makedirs(CFG.output_dir, exist_ok=True)

# Set random seeds
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Using device: {CFG.device}")
print(f"Training {len(CFG.selected_folds)} folds over {len(CFG.seeds)} seeds")

# ============================================================================
# DATASET CLASS
# ============================================================================
class BiomassDataset(Dataset):
    """
    Dataset for training/validation.
    Loads images, tabular features, and target values.
    """
    def __init__(self, df, img_dir, transform=None, tabular_scaler=None, 
                 target_scaler=None, is_training=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_training = is_training

        # Prepare tabular features
        tabular_data = df[['Pre_GSHH_NDVI', 'Height_Ave_cm']].fillna(0).values

        if tabular_scaler is not None:
            self.tabular_features = tabular_scaler.transform(tabular_data)
        else:
            self.tabular_features = tabular_data

        # Prepare targets
        if is_training:
            target_data = df[CFG.targets].values

            if target_scaler is not None:
                self.targets = target_scaler.transform(target_data)
            else:
                self.targets = target_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.img_dir, row['image_path'].split('/')[-1])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Get tabular features
        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        if self.is_training:
            # Get targets
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, tabular, target
        else:
            return image, tabular

# ============================================================================
# AUGMENTATION TRANSFORMS
# ============================================================================
def get_train_transforms(): 
    """Training transforms with augmentation""" 
    return A.Compose([A.HorizontalFlip(p=0.5), 
                      A.VerticalFlip(p=0.5), 
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), 
                      A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5), 
                      A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), 
                      A.Resize(CFG.img_size, CFG.img_size),
                      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(), ])


def get_valid_transforms():
    """Validation transforms (no augmentation)"""
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class BiomassModel(nn.Module):
    """
    Multi-modal model combining image and tabular features.
    """
    def __init__(self, model_name, pretrained=False, backbone_path=None):
        super(BiomassModel, self).__init__()

        # Image encoder
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Manually load weights if provided
        if backbone_path and os.path.exists(backbone_path):
            print(f"Loading offline backbone weights from: {backbone_path}")
            weights = torch.load(backbone_path, map_location='cpu')
            # Determine if we need to clean keys (sometimes timm adds prefixes)
            self.backbone.load_state_dict(weights, strict=False)
        elif pretrained:
            # Fallback for online debugging if needed, though usually better to be explicit
            print("Warning: Attempting to download weights (pretrained=True)")

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, CFG.img_size, CFG.img_size)
            img_features = self.backbone(dummy_input).shape[1]

        # Tabular feature encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Fusion layer
        fusion_dim = img_features + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Output heads (5 targets)
        self.head_green = nn.Linear(256, 1)
        self.head_dead = nn.Linear(256, 1)
        self.head_clover = nn.Linear(256, 1)
        self.head_gdm = nn.Linear(256, 1)
        self.head_total = nn.Linear(256, 1)

    def forward(self, image, tabular):
        # Extract features
        img_features = self.backbone(image)
        tab_features = self.tabular_encoder(tabular)

        # Fuse
        combined = torch.cat([img_features, tab_features], dim=1)
        fused = self.fusion(combined)

        # Predict
        out_green = self.head_green(fused)
        out_dead = self.head_dead(fused)
        out_clover = self.head_clover(fused)
        out_gdm = self.head_gdm(fused)
        out_total = self.head_total(fused)

        outputs = torch.cat([out_green, out_dead, out_clover, out_gdm, out_total], dim=1)
        return outputs

# ============================================================================
# LOSS FUNCTION
# ============================================================================
class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        self.beta = beta 

    def forward(self, pred, target):
        w = self.weights.to(pred.device)

        # Standard SmoothL1 from PyTorch (applied element-wise)
        loss_fn = nn.SmoothL1Loss(reduction='none', beta=self.beta)
        loss = loss_fn(pred, target)

        # Apply weights and mean
        return (loss * w).mean()



# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [TRAIN]')
    for images, tabular, targets in pbar:
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, tabular)
        main_loss = criterion(outputs, targets)
        loss = main_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    return running_loss / len(loader)

def validate(model, loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [VALID]')
    with torch.no_grad():
        for images, tabular, targets in pbar:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            outputs = model(images, tabular)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    return running_loss / len(loader)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train_fold(fold, train_df, val_df, current_seed):
    """Train a single fold"""
    print(f"\n{'='*70}")
    print(f"TRAINING | SEED {current_seed} | FOLD {fold}")
    print(f"{'='*70}")

    # Create scalers
    tabular_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit scalers on training data
    tabular_scaler.fit(train_df[['Pre_GSHH_NDVI', 'Height_Ave_cm']].fillna(0))
    target_scaler.fit(train_df[CFG.targets])

    # Create datasets
    train_dataset = BiomassDataset(
        train_df, CFG.train_dir,
        transform=get_train_transforms() if CFG.use_augmentation else get_valid_transforms(),
        tabular_scaler=tabular_scaler,
        target_scaler=target_scaler,
        is_training=True
    )

    val_dataset = BiomassDataset(
        val_df, CFG.train_dir,
        transform=get_valid_transforms(),
        tabular_scaler=tabular_scaler,
        target_scaler=target_scaler,
        is_training=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True
    )

    # Create model
    model = BiomassModel(CFG.model_name, pretrained=False, backbone_path=CFG.weights_path).to(CFG.device)
    # Freeze early layers
    for name, param in list(model.backbone.named_parameters())[: len(list(model.backbone.named_parameters())) // 2]:
        param.requires_grad = False

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Loss function
    criterion = WeightedSmoothL1Loss()

    # Training loop
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(CFG.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device, epoch)

        # Validate
        val_loss = validate(model, val_loader, criterion, CFG.device, epoch)

        # Scheduler step
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_name = f'best_model_seed{current_seed}_fold{fold}.pth'
            save_path = os.path.join(CFG.output_dir, save_name)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'tabular_scaler': tabular_scaler,
                'target_scaler': target_scaler,
                'epoch': epoch,
                'val_loss': val_loss,
                'seed': current_seed
            }

            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model to {save_name} (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n✓ Seed {current_seed} | Fold {fold} completed. Best Loss: {best_val_loss:.4f}")
    return best_val_loss

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("CSIRO BIOMASS PREDICTION - SEEDED ENSEMBLE TRAINING")
    print("="*70)

    # Load data
    print("\n[1/3] Loading training data...")
    train_df = pd.read_csv(CFG.train_csv)

    # Pivot data
    print("✓ Reshaping data...")
    train_df_pivot = train_df.pivot(
        index=['image_path', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    for target in CFG.targets:
        if target not in train_df_pivot.columns:
            train_df_pivot[target] = 0.0

    print(f"✓ Reshaped to {len(train_df_pivot)} unique images")

    all_results = []

    for seed in CFG.seeds:
        print(f"\n\n{'#'*70}")
        print(f"STARTING RUN WITH SEED: {seed}")
        print(f"{'#'*70}")

        # Set seed for this run
        set_seed(seed)

        # Create K-Fold splits (Shuffle uses current seed)
        print(f"\nCreating K-Fold splits with seed {seed}...")
        kfold = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=seed)

        seed_results = []

        # Train folds for this seed
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df_pivot)):
            if fold not in CFG.selected_folds:
                continue

            train_fold_df = train_df_pivot.iloc[train_idx].reset_index(drop=True)
            val_fold_df = train_df_pivot.iloc[val_idx].reset_index(drop=True)

            # Pass seed to train_fold
            best_loss = train_fold(fold, train_fold_df, val_fold_df, seed)
            seed_results.append(best_loss)

            # Clean up GPU
            torch.cuda.empty_cache()

        avg_seed_loss = np.mean(seed_results)
        all_results.append({'seed': seed, 'avg_loss': avg_seed_loss})
        print(f"\n>>> Seed {seed} Finished. Average Loss: {avg_seed_loss:.4f}")

    # Summary
    print("\n" + "="*70)
    print("ALL SEEDS TRAINING COMPLETE")
    print("="*70)
    for res in all_results:
        print(f"Seed {res['seed']} Avg Loss: {res['avg_loss']:.4f}")

    total_avg = np.mean([r['avg_loss'] for r in all_results])
    print(f"\nOverall Average Loss: {total_avg:.4f}")

if __name__ == '__main__':
    main()
