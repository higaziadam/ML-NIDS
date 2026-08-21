"""
Parquet data explorer with visual output for ML-NIDS Kaggle datasets.

Reads and explores CICIDS2018 parquet files from Kaggle with beautiful formatting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import sys

# Support the documented invocation: ``python scripts/explore_parquet.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import logger


def configure_console_encoding() -> None:
    """Use UTF-8 output when the Windows console defaults to a legacy code page."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ParquetExplorer:
    """Explore parquet files with visual output."""
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize explorer.
        
        Args:
            data_dir: Directory containing parquet files
        """
        if data_dir is None:
            data_dir = Path("data")
        
        self.data_dir = Path(data_dir)
        # Look for parquet files in raw/ subdirectory
        raw_dir = self.data_dir / "raw"
        self.parquet_files = list(raw_dir.glob("*.parquet")) if raw_dir.exists() else list(self.data_dir.glob("*.parquet"))
        self.data_cache = {}
        
        logger.info(f"Found {len(self.parquet_files)} parquet files")
    
    def print_header(self, text: str, width: int = 70) -> None:
        """Print formatted header."""
        print("\n" + "="*width)
        print(f"  {text}")
        print("="*width)
    
    def print_subheader(self, text: str, width: int = 70) -> None:
        """Print formatted subheader."""
        print(f"\n{'─'*width}")
        print(f"  ▸ {text}")
        print(f"{'─'*width}")
    
    def print_row(self, label: str, value: str, width: int = 70) -> None:
        """Print formatted key-value row."""
        label_str = f"  {label}:"
        print(f"{label_str:<35} {value}")
    
    def load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load parquet file with caching."""
        if file_path not in self.data_cache:
            self.data_cache[file_path] = pd.read_parquet(file_path)
        return self.data_cache[file_path]
    
    def print_file_list(self) -> None:
        """Print list of all parquet files."""
        self.print_header("📦 AVAILABLE PARQUET FILES", 70)
        
        for i, file in enumerate(sorted(self.parquet_files), 1):
            size_mb = file.stat().st_size / (1024**2)
            attack_type = file.stem.split("_")[0]
            print(f"  {i:2d}. {attack_type:<20} ({size_mb:>7.2f} MB)")
        
        print("="*70 + "\n")
    
    def print_dataset_summary(self, df: pd.DataFrame, file_name: str) -> None:
        """Print dataset summary."""
        self.print_header(f"📊 DATASET SUMMARY: {file_name}", 70)
        
        self.print_row("Shape", f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        self.print_row("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        self.print_row("Duplicates", f"{df.duplicated().sum():,}")
        self.print_row("Missing values", f"{df.isnull().sum().sum():,}")
        
        print("\n" + "="*70 + "\n")
    
    def print_column_info(self, df: pd.DataFrame) -> None:
        """Print column information."""
        self.print_subheader("📋 Column Information", 70)
        
        print(f"\n{'Column Name':<30} {'Type':<15} {'Non-Null':<12} {'Unique':<10}")
        print("─"*70)
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = f"{df[col].notna().sum():,}"
            unique = f"{df[col].nunique():,}"
            print(f"{col:<30} {col_type:<15} {non_null:<12} {unique:<10}")
        
        print()
    
    def print_numeric_stats(self, df: pd.DataFrame) -> None:
        """Print numeric statistics."""
        self.print_subheader("🔢 Numeric Statistics", 70)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("  No numeric columns found\n")
            return
        
        stats = df[numeric_cols].describe().T
        
        for col in stats.index[:10]:  # First 10 numeric columns
            print(f"\n  {col}:")
            print(f"    Mean:  {stats.loc[col, 'mean']:>15.4f}")
            print(f"    Std:   {stats.loc[col, 'std']:>15.4f}")
            print(f"    Min:   {stats.loc[col, 'min']:>15.4f}")
            print(f"    Max:   {stats.loc[col, 'max']:>15.4f}")
            print(f"    Q1:    {stats.loc[col, '25%']:>15.4f}")
            print(f"    Q3:    {stats.loc[col, '75%']:>15.4f}")
        
        if len(numeric_cols) > 10:
            print(f"\n  ... and {len(numeric_cols) - 10} more numeric columns")
        
        print()
    
    def print_label_distribution(self, df: pd.DataFrame) -> None:
        """Print label distribution if label column exists."""
        # Try to find label column
        label_col = None
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            return
        
        self.print_subheader(f"🎯 Label Distribution ({label_col})", 70)
        
        label_counts = df[label_col].value_counts().sort_values(ascending=False)
        total = len(df)
        
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"  {str(label):<20} {count:>10,} ({percentage:>6.2f}%) {bar}")
        
        print()
    
    def explore_file(self, file_index: int = 0) -> pd.DataFrame:
        """Explore specific parquet file."""
        if file_index >= len(self.parquet_files):
            print(f"Error: File index {file_index} out of range")
            return None
        
        file_path = sorted(self.parquet_files)[file_index]
        df = self.load_parquet(file_path)
        
        print("\n" + "▶"*35 + "\n")
        self.print_dataset_summary(df, file_path.name)
        self.print_column_info(df)
        self.print_numeric_stats(df)
        self.print_label_distribution(df)
        
        return df
    
    def explore_all_files(self) -> Dict[str, pd.DataFrame]:
        """Explore all parquet files with summary."""
        self.print_header("🔍 EXPLORING ALL PARQUET FILES", 70)
        
        all_data = {}
        summaries = []
        
        for file_path in sorted(self.parquet_files):
            df = self.load_parquet(file_path)
            attack_type = file_path.stem.split("_")[0]
            all_data[attack_type] = df
            
            # Get label column
            label_col = None
            for col in df.columns:
                if 'label' in col.lower() or 'class' in col.lower():
                    label_col = col
                    break
            
            label_info = "N/A"
            if label_col:
                top_label = df[label_col].value_counts().index[0]
                label_info = str(top_label)
            
            summaries.append({
                "Attack": attack_type,
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "Memory (MB)": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
                "Primary Label": label_info
            })
        
        summary_df = pd.DataFrame(summaries)
        
        print("\n" + "─"*85)
        print(f"{'Attack Type':<20} {'Rows':>12} {'Columns':>10} {'Memory (MB)':>15} {'Primary Label':<20}")
        print("─"*85)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Attack']:<20} {row['Rows']:>12,} {row['Columns']:>10} {row['Memory (MB)']:>15} {row['Primary Label']:<20}")
        
        print("─"*85)
        print(f"{'TOTAL':<20} {summary_df['Rows'].sum():>12,} samples across {len(all_data)} attack types")
        print("="*85 + "\n")
        
        return all_data
    
    def get_sample_data(self, file_index: int = 0, n_rows: int = 5) -> pd.DataFrame:
        """Get sample data from file."""
        if file_index >= len(self.parquet_files):
            return None
        
        file_path = sorted(self.parquet_files)[file_index]
        df = self.load_parquet(file_path)
        
        return df.head(n_rows)
    
    def combine_all_data(self, add_source_column: bool = True) -> pd.DataFrame:
        """Combine all parquet files into single DataFrame."""
        combined = []
        
        for file_path in sorted(self.parquet_files):
            df = self.load_parquet(file_path)
            
            if add_source_column:
                attack_type = file_path.stem.split("_")[0]
                df["source"] = attack_type
            
            combined.append(df)
        
        combined_df = pd.concat(combined, ignore_index=True)
        
        logger.info(f"Combined dataset: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
        
        return combined_df


def main():
    """Main interactive explorer."""
    configure_console_encoding()
    explorer = ParquetExplorer(data_dir=Path("data"))
    
    # Print available files
    explorer.print_file_list()
    
    # Explore all files with summary
    explorer.explore_all_files()
    
    # Show detailed view of first file
    print("\n" + "▶"*35)
    print("  Detailed view of first dataset:\n")
    first_df = explorer.explore_file(0)
    
    if first_df is not None:
        print("\n📄 Sample Data (First 5 rows):")
        print("─"*100)
        print(first_df.head().to_string())
        print("─"*100 + "\n")


if __name__ == "__main__":
    main()
