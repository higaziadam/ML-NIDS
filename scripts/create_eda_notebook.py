"""
Interactive Jupyter notebook setup for exploring CICIDS2018 data visually.

This script creates a Jupyter notebook for EDA with beautiful visualizations.
"""

import json
from pathlib import Path


def create_eda_notebook():
    """Create Jupyter notebook for EDA."""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🔍 ML-NIDS Data Exploration & Analysis\n",
                    "\n",
                    "## CICIDS2018 Kaggle Dataset Analysis\n",
                    "\n",
                    "This notebook explores and visualizes the CICIDS2018 network traffic dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Imports\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from pathlib import Path\n",
                    "import sys\n",
                    "\n",
                    "sys.path.insert(0, '..')\n",
                    "from scripts.explore_parquet import ParquetExplorer\n",
                    "\n",
                    "# Set style\n",
                    "sns.set_style('darkgrid')\n",
                    "plt.rcParams['figure.figsize'] = (14, 6)\n",
                    "plt.rcParams['font.size'] = 10"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Overview"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "explorer = ParquetExplorer(data_dir=Path('../data'))\n",
                    "explorer.print_file_list()\n",
                    "all_data = explorer.explore_all_files()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Detailed Analysis of First Dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load first dataset\n",
                    "first_df = explorer.explore_file(0)\n",
                    "print(f\"\\nFirst dataset shape: {first_df.shape}\")\n",
                    "print(f\"\\nColumn names:\")\n",
                    "print(first_df.columns.tolist())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Label Distribution Across All Datasets"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load all data\n",
                    "combined_df = explorer.combine_all_data(add_source_column=True)\n",
                    "\n",
                    "print(f\"Combined dataset shape: {combined_df.shape}\")\n",
                    "print(f\"\\nAttack distribution:\")\n",
                    "print(combined_df['source'].value_counts())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Visualize attack distribution\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "\n",
                    "# Bar plot\n",
                    "combined_df['source'].value_counts().plot(\n",
                    "    kind='bar', ax=axes[0], color='steelblue', edgecolor='black'\n",
                    ")\n",
                    "axes[0].set_title('Attack Type Distribution', fontsize=12, fontweight='bold')\n",
                    "axes[0].set_ylabel('Count', fontsize=11)\n",
                    "axes[0].set_xlabel('Attack Type', fontsize=11)\n",
                    "axes[0].tick_params(axis='x', rotation=45)\n",
                    "\n",
                    "# Pie chart\n",
                    "combined_df['source'].value_counts().plot(\n",
                    "    kind='pie', ax=axes[1], autopct='%1.1f%%', colors=sns.color_palette('Set2')\n",
                    ")\n",
                    "axes[1].set_title('Attack Type Proportions', fontsize=12, fontweight='bold')\n",
                    "axes[1].set_ylabel('')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "print(f\"Total samples: {len(combined_df):,}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Feature Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Get numeric columns\n",
                    "numeric_cols = first_df.select_dtypes(include=[np.number]).columns\n",
                    "print(f\"Number of numeric features: {len(numeric_cols)}\")\n",
                    "print(f\"\\nNumeric features:\")\n",
                    "for i, col in enumerate(numeric_cols[:10], 1):\n",
                    "    print(f\"  {i:2d}. {col}\")\n",
                    "\n",
                    "if len(numeric_cols) > 10:\n",
                    "    print(f\"  ... and {len(numeric_cols) - 10} more\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Visualize feature statistics\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "\n",
                    "# Missing values\n",
                    "missing = first_df.isnull().sum()\n",
                    "missing = missing[missing > 0].sort_values(ascending=False)\n",
                    "if len(missing) > 0:\n",
                    "    missing.head(10).plot(kind='barh', ax=axes[0, 0], color='coral')\n",
                    "    axes[0, 0].set_title('Top 10 Columns with Missing Values', fontweight='bold')\n",
                    "    axes[0, 0].set_xlabel('Count')\n",
                    "else:\n",
                    "    axes[0, 0].text(0.5, 0.5, 'No Missing Values! ✓', ha='center', va='center',\n",
                    "                    fontsize=14, transform=axes[0, 0].transAxes)\n",
                    "    axes[0, 0].set_title('Missing Values', fontweight='bold')\n",
                    "\n",
                    "# Data types\n",
                    "dtype_counts = first_df.dtypes.value_counts()\n",
                    "dtype_counts.plot(kind='bar', ax=axes[0, 1], color='lightgreen', edgecolor='black')\n",
                    "axes[0, 1].set_title('Data Type Distribution', fontweight='bold')\n",
                    "axes[0, 1].set_ylabel('Count')\n",
                    "axes[0, 1].set_xlabel('Data Type')\n",
                    "axes[0, 1].tick_params(axis='x', rotation=45)\n",
                    "\n",
                    "# Memory usage\n",
                    "memory = first_df.memory_usage(deep=True) / 1024**2\n",
                    "memory[memory > 0].nlargest(10).plot(kind='barh', ax=axes[1, 0], color='skyblue')\n",
                    "axes[1, 0].set_title('Top 10 Columns by Memory Usage', fontweight='bold')\n",
                    "axes[1, 0].set_xlabel('Memory (MB)')\n",
                    "\n",
                    "# Dataset info\n",
                    "info_text = (\n",
                    "    f\"Dataset Shape: {first_df.shape}\\n\\n\"\n",
                    "    f\"Total Rows: {len(first_df):,}\\n\"\n",
                    "    f\"Total Columns: {len(first_df.columns)}\\n\"\n",
                    "    f\"Total Memory: {first_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\\n\"\n",
                    "    f\"Duplicates: {first_df.duplicated().sum():,}\\n\"\n",
                    "    f\"Missing Values: {first_df.isnull().sum().sum():,}\"\n",
                    ")\n",
                    "axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, family='monospace',\n",
                    "                verticalalignment='center', transform=axes[1, 1].transAxes)\n",
                    "axes[1, 1].axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Numeric Features Statistics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Show statistics\n",
                    "stats = first_df[numeric_cols[:15]].describe().T\n",
                    "print(\"Numeric Features Statistics (First 15):\")\n",
                    "print(stats.to_string())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Distribution plots\n",
                    "n_features = min(8, len(numeric_cols))\n",
                    "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
                    "axes = axes.flatten()\n",
                    "\n",
                    "for i, col in enumerate(numeric_cols[:n_features]):\n",
                    "    axes[i].hist(first_df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)\n",
                    "    axes[i].set_title(f'{col}', fontweight='bold', fontsize=10)\n",
                    "    axes[i].set_ylabel('Frequency')\n",
                    "    axes[i].grid(axis='y', alpha=0.3)\n",
                    "\n",
                    "for i in range(n_features, len(axes)):\n",
                    "    axes[i].axis('off')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Data Preparation for Training"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "from scripts.prepare_kaggle_data import KaggleDataProcessor\n",
                    "\n",
                    "processor = KaggleDataProcessor(data_dir=Path('../data'))\n",
                    "\n",
                    "# Option 1: Quick load and view\n",
                    "print(\"🔄 Loading all parquet files...\")\n",
                    "combined = processor.load_all_parquets()\n",
                    "\n",
                    "print(f\"\\n✓ Combined dataset shape: {combined.shape}\")\n",
                    "print(f\"\\nLabel distribution:\")\n",
                    "print(combined['Label'].value_counts())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Standardize and prepare for training\n",
                    "print(\"\\n📊 Preparing data for training...\")\n",
                    "combined_processed = processor.standardize_labels(combined.copy())\n",
                    "\n",
                    "print(f\"\\nAfter standardization:\")\n",
                    "print(combined_processed['Label'].value_counts())\n",
                    "\n",
                    "# Visualize class distribution\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
                    "\n",
                    "combined_processed['Label'].value_counts().plot(kind='bar', ax=axes[0], \n",
                    "                                                 color=['green', 'red'], \n",
                    "                                                 edgecolor='black')\n",
                    "axes[0].set_title('Binary Classification Distribution', fontweight='bold')\n",
                    "axes[0].set_ylabel('Count')\n",
                    "axes[0].set_xlabel('Class (0=Benign, 1=Attack)')\n",
                    "axes[0].tick_params(axis='x', rotation=0)\n",
                    "\n",
                    "labels = ['Benign (0)', 'Attack (1)']\n",
                    "combined_processed['Label'].value_counts().plot(kind='pie', ax=axes[1], \n",
                    "                                                autopct='%1.1f%%',\n",
                    "                                                labels=labels,\n",
                    "                                                colors=['green', 'red'])\n",
                    "axes[1].set_title('Class Proportions', fontweight='bold')\n",
                    "axes[1].set_ylabel('')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Ready to Train!\n",
                    "\n",
                    "Run the data preparation script to process all data:\n",
                    "\n",
                    "```bash\n",
                    "python scripts/prepare_kaggle_data.py\n",
                    "```\n",
                    "\n",
                    "Then train your model:\n",
                    "\n",
                    "```bash\n",
                    "python src/train.py --data data/processed/train_data.csv --model random_forest\n",
                    "```"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def main():
    """Create notebook."""
    notebook = create_eda_notebook()
    
    output_path = Path("notebooks/EDA_CICIDS2018.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ Created Jupyter notebook: {output_path}")
    print(f"\nTo use it:")
    print(f"  1. Install jupyter: pip install jupyter")
    print(f"  2. Run: jupyter notebook notebooks/EDA_CICIDS2018.ipynb")


if __name__ == "__main__":
    main()
