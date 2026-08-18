"""
Data preparation script for ML-NIDS Kaggle datasets.

Converts parquet files to training-ready format with proper preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import argparse
import sys

# Support the documented invocation: ``python scripts/prepare_kaggle_data.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import logger, Timer, save_data
from src.data_preprocessing import DataCleaner, DataPreprocessor, preprocess_pipeline
from src.config import PROCESSED_DATA_DIR, SPLITS_DIR


class KaggleDataProcessor:
    """Process Kaggle CICIDS2018 datasets."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize processor."""
        self.data_dir = Path(data_dir or "data")
        # Look for parquet files in raw/ subdirectory
        raw_dir = self.data_dir / "raw"
        self.parquet_files = sorted(raw_dir.glob("*.parquet")) if raw_dir.exists() else sorted(self.data_dir.glob("*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {raw_dir if raw_dir.exists() else self.data_dir}."
            )
    
    def load_all_parquets(self) -> pd.DataFrame:
        """Load and combine all parquet files."""
        logger.info(f"Loading {len(self.parquet_files)} parquet files...")
        
        dfs = []
        
        for file_path in self.parquet_files:
            logger.info(f"  Loading {file_path.name}...")
            df = pd.read_parquet(file_path)
            
            # Add attack type
            attack_type = file_path.stem.split("_")[0]
            if 'Label' not in df.columns and 'label' not in df.columns:
                df['Label'] = attack_type
            
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined shape: {combined.shape}")
        
        return combined
    
    def standardize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize label values."""
        logger.info("Standardizing labels...")
        
        # Find label column
        label_col = None
        for col in df.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            logger.warning("No label column found")
            return df
        
        # Convert to binary: Benign=0, Others=1, and expose the stable label
        # column required by the training and validation commands.
        df = df.copy()
        df[label_col] = df[label_col].apply(
            lambda x: 0 if 'benign' in str(x).lower() else 1
        )
        if label_col != "label":
            df = df.rename(columns={label_col: "label"})
        
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df

    def prepare_single_source(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return one labeled source dataset without learned preprocessing.

        Splitting, duplicate removal, feature selection, and scaling are deferred
        to the training or validation workflow. This prevents a preparation step
        from learning information from a final holdout partition.
        """
        if "label" not in df.columns:
            raise ValueError("The standardized dataset must contain a 'label' column.")
        if df["label"].nunique() != 2:
            raise ValueError("The source dataset must contain both benign and attack labels.")
        return df
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        test_split: float = 0.2,
        normalize: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare data for training."""
        logger.info("Preparing data for training...")
        
        # Find label column
        label_col = None
        for col in df.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("No label column found")
        
        # Separate features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        if not 0 < test_split < 1:
            raise ValueError("test_split must be between 0 and 1.")

        # Split before fitting any data-dependent transformation to prevent test
        # samples from influencing cleaning, feature removal, or scaling.
        from sklearn.model_selection import train_test_split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=test_split,
            random_state=42,
            stratify=y,
        )

        with Timer("Data preprocessing"):
            X_train, y_train = preprocess_pipeline(
                X_train_raw, y_train,
                remove_duplicates=True,
                handle_missing="drop",
                # In a supervised NIDS, attacks may be statistical outliers.
                # Removing them can erase an entire class.
                detect_outliers=False,
                remove_constant=True,
                normalize=False,
            )
            X_test_raw = X_test_raw.copy().replace([np.inf, -np.inf], np.nan)
            X_test = DataCleaner.handle_missing_values(X_test_raw, strategy="drop")
            y_test = y_test.loc[X_test.index]
            X_test = X_test.loc[:, X_train.columns]

            if normalize:
                preprocessor = DataPreprocessor(method="minmax")
                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.transform(X_test)
        
        logger.info(f"Train set: {X_train.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def process_and_save(
        self,
        output_dir: Path = None,
        test_split: float = 0.2,
        normalize: bool = True,
        single_source: bool = False,
        output_file: str = "cicids2018_labeled.csv",
        overwrite: bool = False,
    ) -> None:
        """Process all data and save to disk."""
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting full data processing pipeline...")
        
        with Timer("Full pipeline"):
            # Load
            df = self.load_all_parquets()
            
            # Standardize labels
            df = self.standardize_labels(df)
            
            if single_source:
                source = self.prepare_single_source(df)
                source_path = output_dir / output_file
                if source_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Refusing to overwrite existing source dataset: {source_path}. "
                        "Choose a new --output-file or pass --overwrite."
                    )
                save_data(source, source_path)
                logger.info(
                    "Saved one labeled source dataset without learned preprocessing: "
                    f"{source_path}"
                )
                return

            # Legacy preparation path. Prefer --single-source with
            # ``python -m src.validation create-holdout`` for new experiments.
            X_train, X_test, y_train, y_test = self.prepare_for_training(
                df,
                test_split=test_split,
                normalize=normalize,
            )
            
            # Save
            logger.info("Saving processed data...")
            
            # Combine X and y
            train_df = X_train.copy()
            train_df['label'] = y_train.values
            
            test_df = X_test.copy()
            test_df['label'] = y_test.values
            
            # Save to both processed and splits directories
            save_data(train_df, output_dir / "train_data.csv")
            save_data(test_df, output_dir / "test_data.csv")
            save_data(train_df, SPLITS_DIR / "train.csv")
            save_data(test_df, SPLITS_DIR / "test.csv")
            
            logger.info("Data processing complete!")
            logger.info(f"Saved to: {output_dir}")


def main():
    """Main data preparation."""
    parser = argparse.ArgumentParser(description="Prepare Kaggle CICIDS2018 data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory with parquet files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Test set proportion"
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize with parameters fitted only on the training split (default: disabled)"
    )
    parser.add_argument(
        "--single-source",
        action="store_true",
        help="Save one labeled source dataset; defer all learned processing and splitting"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="cicids2018_labeled.csv",
        help="Output filename used with --single-source"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacement of an existing single-source dataset"
    )
    
    args = parser.parse_args()
    
    processor = KaggleDataProcessor(data_dir=args.data_dir)
    processor.process_and_save(
        output_dir=args.output_dir,
        test_split=args.test_split,
        normalize=args.normalize,
        single_source=args.single_source,
        output_file=args.output_file,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
