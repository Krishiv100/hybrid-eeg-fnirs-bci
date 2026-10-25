import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


class EEGBranchNet(nn.Module):
    """
    Compact Spatial-Spectral CNN + MLP branch for EEG features & downsampled temporal profiles.
    Input: (batch_size, 120) -> reshaped to (batch_size, 4 bands, 30 channels)
    """
    def __init__(self, in_dim: int = 120, embed_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 + in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU()
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 120) -> reshape into (B, 4 frequency bands, 30 spatial channels)
        x_spatial = x.view(x.size(0), 4, 30)
        conv_out = self.spatial_conv(x_spatial)
        combined = torch.cat([conv_out, x], dim=1)
        return self.fc(combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        return self.classifier(z)


class fNIRSBranchNet(nn.Module):
    """
    1D Spatiotemporal Hemodynamic CNN + Multi-Window Feature Branch for fNIRS.
    Processes both raw temporal curves (B, 72, 150) and multi-window features (B, 432).
    """
    def __init__(self, feat_dim: int = 432, embed_dim: int = 32, num_classes: int = 2):
        super().__init__()
        # Temporal-Spatial Conv1D on raw (B, 72, 150) hemodynamic response
        self.temp_conv = nn.Sequential(
            nn.Conv1d(in_channels=72, out_channels=32, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 + feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU()
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x_feat: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        conv_out = self.temp_conv(x_raw)
        combined = torch.cat([conv_out, x_feat], dim=1)
        return self.fc(combined)

    def forward(self, x_feat: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x_feat, x_raw)
        return self.classifier(z)


class GatedHybridBCINet(nn.Module):
    """
    Dual-Branch Attention-Gated Multimodal Fusion Network combining EEG and fNIRS.
    Uses a learnable sigmoid attention gate to dynamically weight EEG and fNIRS representations.
    """
    def __init__(self, eeg_dim: int = 120, fnirs_dim: int = 432, embed_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.eeg_branch = EEGBranchNet(in_dim=eeg_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.fnirs_branch = fNIRSBranchNet(feat_dim=fnirs_dim, embed_dim=embed_dim, num_classes=num_classes)

        # Learnable modality attention gate
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 3, num_classes)
        )

    def forward(self, x_eeg: torch.Tensor, x_fnirs_feat: torch.Tensor, x_fnirs_raw: torch.Tensor):
        z_eeg = self.eeg_branch.forward_features(x_eeg)
        z_fnirs = self.fnirs_branch.forward_features(x_fnirs_feat, x_fnirs_raw)

        # Attention-gated representation
        g = self.gate(torch.cat([z_eeg, z_fnirs], dim=1))
        z_gated = g * z_eeg + (1.0 - g) * z_fnirs

        # Concatenate gated representation with individual embeddings for residual skip connection
        z_joint = torch.cat([z_gated, z_eeg, z_fnirs], dim=1)
        logits_joint = self.fusion_head(z_joint)
        logits_eeg = self.eeg_branch.classifier(z_eeg)
        logits_fnirs = self.fnirs_branch.classifier(z_fnirs)
        return logits_joint, logits_eeg, logits_fnirs


def evaluate_deep_subject(
    eeg_feats: np.ndarray,
    fnirs_feats: np.ndarray,
    X_fnirs_raw: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 80,
    lr: float = 3e-3
):
    """
    Trains and evaluates EEGBranchNet, fNIRSBranchNet, and GatedHybridBCINet
    using 5-fold Stratified Cross-Validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_eeg_list, acc_fnirs_list, acc_hybrid_list = [], [], []

    for train_idx, test_idx in cv.split(eeg_feats, y):
        scaler_e = StandardScaler()
        scaler_f = StandardScaler()

        Xe_tr = scaler_e.fit_transform(eeg_feats[train_idx])
        Xe_te = scaler_e.transform(eeg_feats[test_idx])

        Xf_tr = scaler_f.fit_transform(fnirs_feats[train_idx])
        Xf_te = scaler_f.transform(fnirs_feats[test_idx])

        # Standardize raw fNIRS temporal tensor across trials/time
        raw_mean = X_fnirs_raw[train_idx].mean(axis=(0, 2), keepdims=True)
        raw_std = X_fnirs_raw[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8
        Xr_tr = (X_fnirs_raw[train_idx] - raw_mean) / raw_std
        Xr_te = (X_fnirs_raw[test_idx] - raw_mean) / raw_std

        t_Xe_tr = torch.tensor(Xe_tr, dtype=torch.float32, device=device)
        t_Xe_te = torch.tensor(Xe_te, dtype=torch.float32, device=device)
        t_Xf_tr = torch.tensor(Xf_tr, dtype=torch.float32, device=device)
        t_Xf_te = torch.tensor(Xf_te, dtype=torch.float32, device=device)
        t_Xr_tr = torch.tensor(Xr_tr, dtype=torch.float32, device=device)
        t_Xr_te = torch.tensor(Xr_te, dtype=torch.float32, device=device)
        t_y_tr = torch.tensor(y[train_idx], dtype=torch.long, device=device)
        y_te = y[test_idx]

        # Initialize Gated Hybrid Model (trains EEG branch, fNIRS branch, and Joint Fusion simultaneously via multi-task loss)
        model = GatedHybridBCINet(
            eeg_dim=Xe_tr.shape[1],
            fnirs_dim=Xf_tr.shape[1],
            embed_dim=32
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits_joint, logits_eeg, logits_fnirs = model(t_Xe_tr, t_Xf_tr, t_Xr_tr)
            # Multi-task deep supervision loss ensures each branch learns strong independent representations
            loss = criterion(logits_joint, t_y_tr) + 0.5 * criterion(logits_eeg, t_y_tr) + 0.5 * criterion(logits_fnirs, t_y_tr)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_joint, logits_eeg, logits_fnirs = model(t_Xe_te, t_Xf_te, t_Xr_te)
            prob_joint = torch.softmax(logits_joint, dim=1)
            prob_eeg = torch.softmax(logits_eeg, dim=1)
            prob_fnirs = torch.softmax(logits_fnirs, dim=1)

            # Evaluate single branches vs. deep attention-gated ensemble
            pred_eeg = prob_eeg.argmax(dim=1).cpu().numpy()
            pred_fnirs = prob_fnirs.argmax(dim=1).cpu().numpy()

            # Combine joint gated head with branch posteriors
            prob_hybrid = 0.5 * prob_joint + 0.3 * prob_eeg + 0.2 * prob_fnirs
            pred_hybrid = prob_hybrid.argmax(dim=1).cpu().numpy()

            acc_eeg_list.append(np.mean(pred_eeg == y_te) * 100.0)
            acc_fnirs_list.append(np.mean(pred_fnirs == y_te) * 100.0)
            acc_hybrid_list.append(np.mean(pred_hybrid == y_te) * 100.0)

    return {
        "EEG": np.mean(acc_eeg_list),
        "fNIRS": np.mean(acc_fnirs_list),
        "Hybrid": np.mean(acc_hybrid_list)
    }