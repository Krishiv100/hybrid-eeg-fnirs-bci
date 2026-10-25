import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score


def get_base_models():
    """
    Returns standard BCI classification pipelines with z-score scaling and ANOVA F-value
    feature selection to prevent high-dimensional overfitting on 60 trials.
    """
    return {
        "Shrinkage_LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=40)),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=40)),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
        ]),
        "Random_Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=40)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }


def evaluate_subject_modalities(eeg_feats: np.ndarray, fnirs_feats: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Evaluates:
      1. EEG-Only
      2. fNIRS-Only
      3. Hybrid (Balanced Feature-Level + Decision-Level Stacking Fusion)
    strictly inside 5-fold Stratified Cross-Validation.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_base_models()
    results = {}

    for model_name, base_pipe in models.items():
        eeg_SelectK = min(30, eeg_feats.shape[1])
        fnirs_SelectK = min(30, fnirs_feats.shape[1])

        fold_acc_eeg = []
        fold_acc_fnirs = []
        fold_acc_hybrid = []

        for train_idx, test_idx in cv.split(eeg_feats, y):
            X_e_tr, X_e_te = eeg_feats[train_idx], eeg_feats[test_idx]
            X_f_tr, X_f_te = fnirs_feats[train_idx], fnirs_feats[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # 1. Fit & evaluate EEG-only branch
            pipe_e = get_base_models()[model_name]
            pipe_e.set_params(select__k=eeg_SelectK)
            pipe_e.fit(X_e_tr, y_tr)
            pred_e = pipe_e.predict(X_e_te)
            prob_e_te = pipe_e.predict_proba(X_e_te)[:, 1:2]
            fold_acc_eeg.append(accuracy_score(y_te, pred_e))

            # 2. Fit & evaluate fNIRS-only branch
            pipe_f = get_base_models()[model_name]
            pipe_f.set_params(select__k=fnirs_SelectK)
            pipe_f.fit(X_f_tr, y_tr)
            pred_f = pipe_f.predict(X_f_te)
            prob_f_te = pipe_f.predict_proba(X_f_te)[:, 1:2]
            fold_acc_fnirs.append(accuracy_score(y_te, pred_f))

            # 3. Hybrid Fusion: Balanced Top-K Feature Concatenation + Out-of-Fold Probability Stacking
            #    Extract top-k scaled features from each modality so neither modality dominates
            e_tr_sel = pipe_e.named_steps["select"].transform(pipe_e.named_steps["scaler"].transform(X_e_tr))
            e_te_sel = pipe_e.named_steps["select"].transform(pipe_e.named_steps["scaler"].transform(X_e_te))
            f_tr_sel = pipe_f.named_steps["select"].transform(pipe_f.named_steps["scaler"].transform(X_f_tr))
            f_te_sel = pipe_f.named_steps["select"].transform(pipe_f.named_steps["scaler"].transform(X_f_te))

            X_hyb_tr = np.concatenate([e_tr_sel, f_tr_sel], axis=1)
            X_hyb_te = np.concatenate([e_te_sel, f_te_sel], axis=1)

            clf_hyb = get_base_models()[model_name].named_steps["clf"]
            clf_hyb.fit(X_hyb_tr, y_tr)
            prob_hyb_te = clf_hyb.predict_proba(X_hyb_te)[:, 1:2]

            # Estimate single-modality CV reliability on the training fold to weight the fusion
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            w_e = max(0.05, cross_val_score(pipe_e, X_e_tr, y_tr, cv=inner_cv).mean() - 0.45)
            w_f = max(0.05, cross_val_score(pipe_f, X_f_tr, y_tr, cv=inner_cv).mean() - 0.45)
            w_h = max(0.05, (w_e + w_f) * 0.6)

            fused_prob = (w_e * prob_e_te + w_f * prob_f_te + w_h * prob_hyb_te) / (w_e + w_f + w_h)
            pred_hyb = (fused_prob.ravel() >= 0.5).astype(np.int64)
            fold_acc_hybrid.append(accuracy_score(y_te, pred_hyb))

        results[model_name] = {
            "EEG": np.mean(fold_acc_eeg) * 100.0,
            "fNIRS": np.mean(fold_acc_fnirs) * 100.0,
            "Hybrid": np.mean(fold_acc_hybrid) * 100.0
        }

    return results