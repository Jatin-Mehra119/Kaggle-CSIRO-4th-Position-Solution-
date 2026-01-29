import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class CFG:
    train_csv = 'org/train.csv'  
    train_dir = 'org/train'      
    output_dir = 'Models_Aux_Only_v7'
    
    weights_path = "vithugedinov3-manual-data-cleaning/best_model_seed42_fold0.pth"

    model_name = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    img_size = 800
    n_folds = 5
    selected_folds = [0, 1, 2, 3, 4]

    epochs = 25 
    batch_size = 8
    learning_rate = 5e-5
    weight_decay = 1e-3 
    num_workers = 16

    seeds = [44]
    # Targets retained for dataset loading structure, though model ignores them
    targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(CFG.output_dir, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# DATASET CLASS
# ============================================================================
class BiomassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, tabular_scaler=None, target_scaler=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        # Tabular data (NDVI, Height) - This is the target for this training
        tabular_data = df[['Pre_GSHH_NDVI', 'Height_Ave_cm']].fillna(0).values
        self.tabular_features = tabular_scaler.transform(tabular_data) if tabular_scaler else tabular_data

        # Main Targets (Loaded to keep structure, but ignored)
        target_data = df[CFG.targets].values
        self.targets = target_scaler.transform(target_data) if target_scaler else target_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'].split('/')[-1])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return image, tabular, target

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class BiomassModel(nn.Module):
    def __init__(self, model_name, backbone_path=None):
        super(BiomassModel, self).__init__()
        # Create raw backbone
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        if backbone_path and os.path.exists(backbone_path):
            print(f"Loading weights from: {backbone_path}")
            checkpoint = torch.load(backbone_path, map_location='cpu', weights_only=False)
            
            # 1. Unwrap dictionary if necessary
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 2. Clean keys (remove 'backbone.' prefix, ignore old heads)
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    # Strip 'backbone.' so it fits into self.backbone directly
                    clean_state_dict[k.replace('backbone.', '')] = v
                elif not k.startswith('biomass_head') and not k.startswith('aux_head'):
                    # Keep raw keys if they aren't head weights
                    clean_state_dict[k] = v
                    
            # 3. Load weights
            msg = self.backbone.load_state_dict(clean_state_dict, strict=False)
            print(f"Weights Loaded. Missing keys (expected for new aux_head): {len(msg.missing_keys)}")

        img_features = self.backbone.num_features

        # AUXILIARY HEAD ONLY (NDVI, Height)
        self.aux_head = nn.Sequential(
            nn.Linear(img_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2) 
        )

    def forward(self, image):
        feat = self.backbone(image)
        aux_out = self.aux_head(feat)
        return aux_out

# ============================================================================
# TRAINING HELPERS
# ============================================================================
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Resize(CFG.img_size, CFG.img_size),
            A.ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training')
    
    # Ignore the 3rd item (biomass targets)
    for img, aux_true, _ in pbar:
        img, aux_true = img.to(device), aux_true.to(device)
        
        optimizer.zero_grad()
        aux_pred = model(img)
        loss = criterion(aux_pred, aux_true)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': total_loss/(pbar.n+1)})
        
    return total_loss/len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, aux_true, _ in loader:
            img, aux_true = img.to(device), aux_true.to(device)
            aux_pred = model(img)
            loss = criterion(aux_pred, aux_true)
            total_loss += loss.item()
            
    return total_loss/len(loader)

# ============================================================================
# MAIN FOLD TRAINING
# ============================================================================
def train_fold(fold, train_df, val_df, seed):
    print(f"\nSEED {seed} | FOLD {fold}")
    
    tab_scaler = StandardScaler().fit(train_df[['Pre_GSHH_NDVI', 'Height_Ave_cm']].fillna(0))
    tar_scaler = StandardScaler().fit(train_df[CFG.targets]) # Maintained for compatibility

    train_ds = BiomassDataset(train_df, CFG.train_dir, get_transforms(True), tab_scaler, tar_scaler)
    val_ds = BiomassDataset(val_df, CFG.train_dir, get_transforms(False), tab_scaler, tar_scaler)

    loader_args = {'batch_size': CFG.batch_size, 'num_workers': CFG.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = BiomassModel(CFG.model_name, CFG.weights_path).to(CFG.device)
    print(model)
    # Freeze 25% of backbone
    params = list(model.backbone.parameters())
    for param in params[:len(params)//4]: param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    # Regression Loss
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience, counter = 8, 0

    for epoch in range(CFG.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        valid_loss = validate(model, val_loader, criterion, CFG.device)
        
        scheduler.step(valid_loss)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            print(f"--> Loss improved from {best_loss:.4f} to {valid_loss:.4f}. Saving model...")
            best_loss = valid_loss
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(), 
                'tab_scaler': tab_scaler, 
                'epoch': epoch
            }, os.path.join(CFG.output_dir, f'best_aux_only_seed{seed}_fold{fold}.pth'))
        else:
            counter += 1
            if counter >= patience: 
                print("Early stopping triggered.")
                break

    return best_loss

def main():
    print("="*70)
    print("CSIRO - AUXILIARY PREDICTION (Using Pretrained Biomass Backbone)")
    print("="*70)
    
    train_df = pd.read_csv(CFG.train_csv)
    train_df_pivot = train_df.pivot(
        index=['image_path', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()
    
    for target in CFG.targets:
        if target not in train_df_pivot.columns:
            train_df_pivot[target] = 0.0
            
    all_results = []
    
    for seed in CFG.seeds:
        set_seed(seed)
        kfold = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=seed)
        seed_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df_pivot)):
            if fold not in CFG.selected_folds:
                continue
            
            train_fold_df = train_df_pivot.iloc[train_idx].reset_index(drop=True)
            val_fold_df = train_df_pivot.iloc[val_idx].reset_index(drop=True)
            
            best_loss = train_fold(fold, train_fold_df, val_fold_df, seed)
            seed_results.append(best_loss)
            torch.cuda.empty_cache()
        
        avg_seed_loss = np.mean(seed_results)
        all_results.append({'seed': seed, 'avg_loss': avg_seed_loss})

    total_avg = np.mean([r['avg_loss'] for r in all_results])
    print(f"\nOverall Average Loss: {total_avg:.4f}")

if __name__ == '__main__':
    main()