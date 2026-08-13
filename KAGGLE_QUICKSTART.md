"""
Quick start guide for using Kaggle CICIDS2018 data with ML-NIDS.
"""

# ============================================================================
# KAGGLE DATA QUICKSTART
# ============================================================================

## 🎯 What You Have

Your ML-NIDS project now includes 10 CICIDS2018 parquet files from Kaggle:
- **Botnet** attacks
- **Bruteforce** attacks  
- **DDoS** attacks (2 scenarios)
- **DoS** attacks (2 scenarios)
- **Infiltration** attacks (2 scenarios)
- **Web** attacks (2 scenarios)

Each file contains network traffic flows with ~80 features (packet/flow statistics)


## 📦 Reading Parquet Files

### Option 1: Simple pandas
```python
import pandas as pd

# Read single file
df = pd.read_parquet("data/Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet")
print(df.shape)
print(df.columns)
print(df.head())
```

### Option 2: Using built-in explorer (VISUALLY APPEALING!)
```bash
python scripts/explore_parquet.py
```

This gives you beautiful formatted output:
- 📦 List of all parquet files with sizes
- 📊 Dataset summaries
- 📋 Column information  
- 🔢 Numeric statistics
- 🎯 Label distributions


## 🔄 Data Processing Workflow

### Step 1: Explore the Data
```bash
python scripts/explore_parquet.py
```

Output example:
```
══════════════════════════════════════════════════════════════════
  📦 AVAILABLE PARQUET FILES
══════════════════════════════════════════════════════════════════
  1. Botnet              (   45.32 MB)
  2. Bruteforce          (   13.25 MB)
  3. DDoS1               (   52.18 MB)
  ... and 7 more files
```

### Step 2: Prepare Data for Training
```bash
python scripts/prepare_kaggle_data.py
```

This will:
1. ✓ Load all 10 parquet files
2. ✓ Combine them into one dataset
3. ✓ Standardize labels (Benign=0, Attack=1)
4. ✓ Preprocess (remove duplicates, handle missing values, normalize)
5. ✓ Split into train/test (80/20)
6. ✓ Save processed data to `data/processed/` and `data/splits/`

Output: 
- `data/processed/train_data.csv` - Training set (~80%)
- `data/processed/test_data.csv` - Test set (~20%)


### Step 3: Create Interactive Jupyter Notebook
```bash
python scripts/create_eda_notebook.py
jupyter notebook notebooks/EDA_CICIDS2018.ipynb
```

This creates a beautiful interactive notebook with:
- 📊 Dataset overview tables
- 📈 Attack distribution visualizations
- 🔢 Feature statistics & distributions
- 🎯 Class imbalance analysis
- Ready-to-use code snippets


### Step 4: Train Your Model
```bash
python src/train.py --data data/processed/train_data.csv --model random_forest --name cicids_model
```

This will:
- Load and preprocess training data
- Engineer features
- Train Random Forest model
- Evaluate on test set
- Save model to `models/saved/cicids_model.pkl`

Expected output:
```
==================================================
MODEL EVALUATION METRICS
==================================================
Accuracy ............... 0.9850
Precision .............. 0.9720
Recall ................. 0.9810
F1-Score ............... 0.9760
ROC-AUC ................ 0.9950
```


### Step 5: Make Predictions
```bash
python src/predict.py --model models/saved/cicids_model.pkl --data data/processed/test_data.csv --output predictions.csv
```


## 📊 Complete Example Pipeline

```bash
# SETUP: All Kaggle parquet files are in data/raw/
ls data/raw/*.parquet  # Should show 10 files

# 1. Explore data visually
python scripts/explore_parquet.py

# 2. Prepare all Kaggle data for training (converts parquets to CSVs)
python scripts/prepare_kaggle_data.py
# Output: data/processed/train_data.csv, test_data.csv

# 3. Train model
python src/train.py \
  --data data/processed/train_data.csv \
  --model random_forest \
  --name cicids_model \
  --test-size 0.2

# 4. Make predictions on test set
python src/predict.py \
  --model models/saved/cicids_model.pkl \
  --data data/processed/test_data.csv \
  --output predictions.csv

# 5. View predictions
head predictions.csv
```


## 🎨 Visual Tools Created

### 1. **explore_parquet.py** - Beautiful Data Explorer
- Formatted tables with UTF-8 box drawing
- Statistics for each dataset
- Column information
- Label distributions
- Sample data preview

Usage:
```python
from scripts.explore_parquet import ParquetExplorer
explorer = ParquetExplorer()
explorer.print_file_list()
all_data = explorer.explore_all_files()
df = explorer.explore_file(0)  # Explore first file
combined = explorer.combine_all_data()  # Combine all
```

### 2. **prepare_kaggle_data.py** - Smart Data Processor
- Loads all 10 parquet files
- Combines with proper attack labeling
- Cleans & normalizes
- Splits train/test
- Saves ready-for-training datasets

Usage:
```bash
python scripts/prepare_kaggle_data.py --test-split 0.2
```

### 3. **create_eda_notebook.py** - Interactive Jupyter Notebook
- Beautiful visualizations
- Attack distribution charts
- Feature statistics
- Data type analysis
- Memory usage profiling
- Ready-to-run code cells

Usage:
```bash
python scripts/create_eda_notebook.py
jupyter notebook notebooks/EDA_CICIDS2018.ipynb
```


## 📝 Features in CICIDS2018

The dataset contains network flow statistics like:
- **Flow Duration** - Length of flow
- **Packets Forward/Backward** - Packet counts
- **Bytes Forward/Backward** - Data volume
- **Flow IAT** - Inter-arrival times
- **Protocol** - TCP/UDP/etc
- **Port numbers** - Source/destination ports
- **Flags** - TCP flags
- **Window sizes** - TCP window
- **Checksum errors** - Error counts
- **Label** - Benign or Attack type

Total: ~80 features per flow record


## 🚀 Recommended Next Steps

1. **Explore the data** - Run `explore_parquet.py` to understand what you have
2. **Create Jupyter notebook** - Run `create_eda_notebook.py` for interactive exploration
3. **Prepare data** - Run `prepare_kaggle_data.py` to process everything
4. **Train model** - Run `src/train.py` with your preferred model
5. **Evaluate** - Check the evaluation metrics and confusion matrix
6. **Deploy** - Use `src/predict.py` for inference on new data


## 📚 Available Models

Your ML-NIDS supports multiple models:
- `random_forest` - Fast, good baseline (recommended)
- `gradient_boosting` - More accurate but slower
- `svm` - Traditional ML approach

Example:
```bash
python src/train.py --model random_forest --name nids_rf
python src/train.py --model gradient_boosting --name nids_gb
python src/train.py --model svm --name nids_svm
```


## 💡 Tips

1. **Start Small**: Test with one parquet file first
   ```python
   df = pd.read_parquet("data/Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet")
   ```

2. **Check Data Size**: Before combining all files
   ```bash
   ls -lh data/*.parquet
   ```

3. **Use Jupyter for Exploration**: More interactive than terminal
   ```bash
   jupyter notebook
   ```

4. **Handle Class Imbalance**: CICIDS2018 is heavily imbalanced (mostly benign)
   - Use `class_weight='balanced'` in config
   - Or use SMOTE in preprocessing

5. **Feature Selection**: With 80+ features, select the most important
   ```python
   from src.feature_extraction import FeatureSelector
   selector = FeatureSelector(n_features=20)
   ```


## 🔗 Links

- CICIDS2018 Kaggle: https://www.kaggle.com/cicdataset/cicids2018
- ML-NIDS Repo: Your GitHub repo
- Pandas Parquet: https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html


## ❓ Troubleshooting

**Q: Memory error when loading all files**
A: Load in batches or increase RAM. Use generator-based loading.

**Q: Missing values in data**
A: Run with `--handle-missing drop` or `mean`

**Q: Class imbalance issues**
A: Use SMOTE balancing in `prepare_kaggle_data.py`

**Q: Parquet file is corrupted**
A: Try re-downloading from Kaggle

---

Happy analyzing! 🎉
