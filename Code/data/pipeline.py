import torch
import pandas as pd
from typing import Tuple, List, Optional
import os
import joblib
from .data_loader import make_loader

from ..models.machine_learning.preprocessing.imputation.imputation import (
    SimpleImputer, MissingIndicator
)
from ..models.machine_learning.preprocessing.encoding.encoding import (
    OneHotEncoder, LabelEncoder
)
from ..models.machine_learning.preprocessing.normalizer.normalizer import StandardScaler
from .clean_tabular import clean_tabular

class TabularPipeline:
    """
    End-to-End Pipeline for cleaning, imputing, and encoding tabular data.
    Builds on Phase 2 Section 11 specifications.
    """
    def __init__(
        self,
        num_cols: List[str],
        cat_cols: List[str],
        target_col: Optional[str] = None,
        imputation_strategy: str = "median",
    ):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.indicator = MissingIndicator(features="missing-only")
        self.encoder = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.is_fitted = False

    def fit_transform(self, df_raw: pd.DataFrame, batch_size=64, val_split=0.2):
        df = clean_tabular(df_raw)
        
        # Verify columns exist
        missing_num = [c for c in self.num_cols if c not in df.columns]
        missing_cat = [c for c in self.cat_cols if c not in df.columns]
        if missing_num or missing_cat:
            raise ValueError(f"Missing expected columns in Dataframe. Num: {missing_num}, Cat: {missing_cat}")

        X_num_df = df[self.num_cols]
        X_cat_df = df[self.cat_cols]
        y = None
        if self.target_col and self.target_col in df.columns:
            y = torch.tensor(df[self.target_col].values, dtype=torch.long)
            
        X_num_tensor = torch.tensor(X_num_df.values, dtype=torch.float32)

        # Handle missing 
        X_flags = self.indicator.fit_transform(X_num_tensor)
        X_num_imp = self.imputer.fit_transform(X_num_tensor)
        
        # Scale
        X_num_sc = self.scaler.fit_transform(X_num_imp)
        
        # Enroll categories via robust label mapping initially before passing to OHE
        X_cat_np = X_cat_df.apply(lambda c: pd.factorize(c)[0]).values
        X_cat_enc = self.encoder.fit_transform(torch.tensor(X_cat_np, dtype=torch.long))

        X_final = torch.cat([X_num_sc, X_flags.float(), X_cat_enc], dim=1)
        
        self.is_fitted = True

        return make_loader(X_final, targets=y, batch_size=batch_size, val_split=val_split, shuffle=True)

    def transform(self, df_raw: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fit before transform!")
        
        df = clean_tabular(df_raw)
        X_num_df = df[self.num_cols]
        X_cat_df = df[self.cat_cols]

        X_num_tensor = torch.tensor(X_num_df.values, dtype=torch.float32)
        X_flags = self.indicator.transform(X_num_tensor)
        X_num_imp = self.imputer.transform(X_num_tensor)
        X_num_sc = self.scaler.transform(X_num_imp)
        
        X_cat_np = X_cat_df.apply(lambda c: pd.factorize(c)[0]).values
        X_cat_enc = self.encoder.transform(torch.tensor(X_cat_np, dtype=torch.long))

        X_final = torch.cat([X_num_sc, X_flags.float(), X_cat_enc], dim=1)
        return X_final

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "target_col": self.target_col,
            "indicator": self.indicator,
            "imputers": self.imputer,
            "encoder": self.encoder,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TabularPipeline":
        state = joblib.load(path)
        pipe = cls(
            num_cols=state["num_cols"],
            cat_cols=state["cat_cols"],
            target_col=state["target_col"]
        )
        pipe.indicator = state["indicator"]
        pipe.imputer = state["imputers"]
        pipe.encoder = state["encoder"]
        pipe.scaler = state["scaler"]
        pipe.is_fitted = state["is_fitted"]
        return pipe
