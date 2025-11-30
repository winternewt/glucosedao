# Gluformer Inference Pipelines: Legacy vs Refactored

## Critical Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `input_chunk_length` | 96 | 8 hours of history (96 × 5min) |
| `output_chunk_length` | 12 | 1 hour prediction (12 × 5min) |
| `num_samples` | 10 (inference) / 100 (training) | Stochastic forward passes for uncertainty |
| `batch_size` | 16 | Process 16 windows simultaneously |
| `length_segment` | 13 | Val/Test segment size in old pipeline |
| `max_length_input` | 192 | Max lookback window |
| `r_drop` | 0.2 | Dropout rate for uncertainty estimation |

## Data Split Ratios (Old Pipeline Only)

```
Total Data: 3287 points (100%)
│
├─ Test OOD: 519 points (15.79%)
│  └─ 10% of subjects (ALL their data)
│     Purpose: Out-of-distribution evaluation
│
└─ Remaining subjects (90%): 2768 points
   │
   ├─ Train: 1948 points (59.26%) ← DISCARDED in inference
   │  └─ Earlier time periods
   │
   ├─ Val: 410 points (12.47%) ← DISCARDED in inference
   │  └─ Middle time periods
   │
   └─ Test: 410 points (12.47%) ← USED for inference (88% data loss!)
      └─ Later time periods
```

---

## Legacy Pipeline (Old Implementation)

### Architecture
**File Flow:** `app.py` → `tools.py` → `utils/darts_processing.py` → `data_formatter/base.py` → `data_formatter/utils.py`

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER UPLOAD CSV                                          │
│    Format: id, time, gl                                     │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. input_processing.py::process_csv_with_options()         │
│    - Auto-detect format (Dexcom/Libre/Unified)             │
│    - Time filtering (min 1 min gap)                        │
│    - Remove duplicates                                      │
│    - Chunking by ID (1000 points/chunk)                    │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. utils/darts_processing.py::load_data()                  │
│    → DataFormatter.__init__()                              │
│      - Load config.yaml                                     │
│      - Encode datetime (day, month, year, hour, min, sec)  │
│      - Interpolate gaps (threshold=45min)                  │
│      - Create continuous segments                          │
│    → DataFormatter.__split_data()                          │
│      - data_formatter/utils.py::split()                    │
│      - Splits into Train/Val/Test/Test OOD                 │
│    ⚠️  PROBLEM: 88% of data discarded here                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Convert to Darts TimeSeries                             │
│    series = {                                               │
│      'train': {'target': [...], 'future': [...]}           │
│      'val': {...}                                           │
│      'test': {...}      ← Only this used                   │
│      'test_ood': {...}                                      │
│    }                                                         │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ScalerCustom.fit_transform()                            │
│    - Fit MinMax scaler on TRAIN split                      │
│    - Transform all splits                                   │
│    ⚠️  Scaler fit on discarded data                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SamplingDatasetInferenceDual(                           │
│      target_series=series['test']['target'],               │
│      covariates=series['test']['future'],                  │
│      input_chunk_length=96,                                │
│      output_chunk_length=12                                │
│    )                                                        │
│    → Only ~410 points → ~136 inference samples             │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. gluformer/model.py::Gluformer.predict()                │
│    - num_samples=10 stochastic forward passes              │
│    - Output: [n_samples, 12, 10]                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Drawbacks

1. **Massive Data Loss**: Users upload days of data, receive predictions for ~12% of it
2. **Unnecessary Computation**: Processes 100%, uses a fraction
3. **Rigid Configuration**: Tightly coupled to `config.yaml` designed for training
4. **Misleading Scaler**: Fits on Train split that doesn't represent user's data distribution

### Data Split Logic (data_formatter/utils.py::split())

**Phase 1: OOD Subject Split**
```python
test_ids = np.random.choice(ids, math.ceil(len(ids) * test_percent_subjects), replace=False)
test_idx_ood = list(df[df[id_col].isin(test_ids)].index)
# Result: 10% of subjects → Test OOD (519 points)
```

**Phase 2: Temporal Split (per remaining subject)**
```python
for each subject:
    segments = get_continuous_time_segments()
    if segments >= 2:
        if last_segment.length >= max_length_input + 3 * length_segment:
            train: all but last (2 * length_segment) points
            val: next length_segment + max_length_input points
            test: last length_segment + max_length_input points
        # ... fallback logic for shorter segments
```

**Characteristics:**
- Train: Earlier time periods within each subject
- Val: Middle periods (for hyperparameter tuning)
- Test: Later periods (temporal continuity maintained)
- Test OOD: Different subjects entirely (generalization test)

---

## Refactored Pipeline (New Implementation)

### Architecture
**File Flow:** `app.py` → `tools.py` → `fast_inference.py` → `cgm_format.FormatProcessor`

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER UPLOAD CSV                                          │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. cgm_format.FormatProcessor                              │
│    - Parse CSV (auto-detect Dexcom/Libre/Unified)         │
│    - Validate schema                                        │
│    - Interpolate gaps (time-weighted linear)               │
│    - Add quality flags (IMPUTATION, CALIBRATION)           │
│    - Convert to unified format                             │
│    ✅ Uses 100% of data                                    │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. fast_inference.py::create_inference_dataset_fast()     │
│    GluformerInferenceConfig(                               │
│      input_chunk_length=96,                                │
│      output_chunk_length=12,                               │
│      datetime_features=['day', 'month', 'year', ...]       │
│    )                                                        │
│    → Uses formatter_utils.interpolate() for compatibility  │
│    → Uses formatter_utils.encode() for datetime features   │
│    → NO SPLITTING: processes entire DataFrame              │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Convert ALL data to Darts TimeSeries                    │
│    - Split by subject ID                                    │
│    - Create target (glucose) series                        │
│    - Create future covariates (datetime features)          │
│    - Maintain static covariates (subject ID)               │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Global Scaling                                           │
│    - Fit MinMax scaler on ENTIRE input dataset             │
│    - Or accept pre-fitted scalers from training            │
│    ✅ Scaler represents actual user data                   │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SamplingDatasetInferenceDual on ALL data               │
│    → 3287 points → ~1000+ inference samples                │
│    ✅ Every valid 108-point window generates prediction    │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. gluformer/model.py::Gluformer.predict()                │
│    - num_samples=10                                         │
│    - Output: [all_samples, 12, 10]                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Improvements

1. **100% Data Utilization**: Every valid window in uploaded file generates prediction
2. **Streamlined Architecture**: Bypasses complex `DataFormatter` class hierarchy
3. **Robust Gap Handling**: `FormatProcessor` provides professional-grade interpolation
4. **Quality Feedback**: UI warnings for gaps, calibration issues, insufficient duration
5. **Decoupled from Training**: No dependency on `config.yaml` training configuration

### Core Function: create_inference_dataset_fast()

**File:** `fast_inference.py`

```python
def create_inference_dataset_fast(
    data: Union[pd.DataFrame, Path, str],
    config: GluformerInferenceConfig,
    scalers: Optional[Dict[str, MinMaxScaler]] = None,
    id_col: str = "id",
    time_col: str = "time",
    target_col: str = "gl"
) -> Tuple[SamplingDatasetInferenceDual, Dict[str, MinMaxScaler], pd.DataFrame]:
    """
    Direct path: DataFrame → Interpolate → Encode → Scale → Darts → Dataset
    No train/val/test splitting. Uses entire input for inference.
    """
```

**Process:**
1. Load DataFrame (if path provided)
2. Reuse `formatter_utils.interpolate()` for compatibility
3. Reuse `formatter_utils.encode()` for datetime features
4. Split by subject ID into continuous segments
5. Convert to Darts TimeSeries (target + future_covariates)
6. Fit/apply MinMax scaling globally
7. Create `SamplingDatasetInferenceDual` from all data

---

## Mathematical Equivalence Proof

**Challenge:** Initial comparison showed index shift and value mismatch.

**Root Cause:** Boundary interpolation difference
- **Old Pipeline**: Interpolated entire file → then sliced Test split
  - First Test point often interpolated using a raw point just before the split boundary
- **New Pipeline (Naive)**: When fed only Test split raw data, lacked that context point
  - Different interpolation at boundary

**Solution (proof_pipeline.py):**
```python
# Identify exact raw time range from old Test split
old_test_start = old_test_data.index.min()
old_test_end = old_test_data.index.max()

# Add ONE preceding raw data point for interpolation context
context_point = raw_data[raw_data['time'] < old_test_start].iloc[-1]
new_input = pd.concat([context_point, test_raw_data])

# Run new pipeline
new_result = create_inference_dataset_fast(new_input, ...)
```

**Verification:**
- Shape match: ✅ Identical
- Value match: ✅ 1e-16 precision (floating point equality)

**Conclusion:** Transformation logic `Interpolate → Encode → Scale → TimeSeries` in `fast_inference.py` is functionally identical to legacy `DataFormatter`.

---

## Dataset Indexing & Sampling

### Valid Sampling Locations (utils/darts_dataset.py)

```python
def get_valid_sampling_locations(target_series, input_chunk_length=96, 
                                 output_chunk_length=12):
    """
    For each time series, find all positions where we can sample 
    108 consecutive points (96 input + 12 output).
    """
    total_length = input_chunk_length + output_chunk_length  # 108
    valid_sampling_locations = {}
    
    for id, series in enumerate(target_series):
        num_entries = len(series)
        if num_entries >= total_length:
            # All valid starting positions
            valid_sampling_locations[id] = list(range(num_entries - total_length + 1))
    
    return valid_sampling_locations
```

**Example:**
- Time series with 205 points
- Valid positions: 0, 1, 2, ..., 97 (98 total)
- Each position = one inference sample

### SamplingDatasetInferenceDual.__getitem__(idx)

```python
def __getitem__(self, idx: int):
    # Map linear index to (series_id, position)
    target_idx = 0
    while idx >= len(self.valid_sampling_locations[target_idx]):
        idx -= len(self.valid_sampling_locations[target_idx])
        target_idx += 1
    
    sampling_location = self.valid_sampling_locations[target_idx][idx]
    
    # Extract windows
    past_target = self.target_series[target_idx][sampling_location:sampling_location+96]
    historic_future_covs = self.covariates[target_idx][sampling_location:sampling_location+96]
    future_covariates = self.covariates[target_idx][sampling_location+96:sampling_location+108]
    static_covariates = self.target_series[target_idx].static_covariates_values()
    
    return past_target, historic_future_covs, future_covariates, static_covariates
```

**Slider Index Example:**
```
Total: 136 samples across 2 series
- Series 0: 43 samples (indices 0-42)
- Series 1: 93 samples (indices 43-135)

User selects index=50:
  → idx=50 >= 43, so use Series 1
  → Position within Series 1: 50 - 43 = 7
  → Extract:
      input_glucose: series_1[7:103]      # 96 points
      input_features: features_1[7:103]   # 96 points
      future_features: features_1[103:115] # 12 points
      ground_truth: series_1[103:115]     # 12 points (for comparison)
```

---

## num_samples: Monte Carlo Dropout

### The Parameter's Role

**Controls:** Number of stochastic forward passes per prediction

**Purpose:** Uncertainty quantification through Monte Carlo Dropout (Gal & Ghahramani, 2016)

### Why Training Mode During Inference?

**File:** `gluformer/model.py::Gluformer.predict()`

```python
def predict(self, test_dataset, batch_size=32, num_samples=100, device='cuda'):
    collate_fn_custom = modify_collate(num_samples)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate_fn_custom)
    
    self.train()  # ← NOT self.eval()
    
    # CRITICAL: This does NOT train the model!
    # - No backpropagation
    # - No gradient computation (outputs detached)
    # - No optimizer steps
    # - Weights remain frozen
    # - ONLY activates dropout for uncertainty estimation
```

### How It Works

**Step 1: Repeat inputs (gluformer/utils/training.py::modify_collate)**
```python
def modify_collate(num_samples):
    def collate_fn_custom(batch):
        # Repeats each batch element num_samples times
        batch_data = default_collate(batch)
        return tuple([torch.cat([x] * num_samples, dim=0) for x in batch_data])
    return collate_fn_custom
```

**Step 2: Multiple forward passes with different dropout masks**
```python
# Input: [batch_size, 96, 1] → Repeated → [batch_size * num_samples, 96, 1]
# Each of num_samples copies gets different dropout mask (r_drop=0.2)
# Output: [batch_size * num_samples, 12, 1]
```

**Step 3: Reshape to distribution**
```python
# Reshape: [batch_size * num_samples, 12, 1] → [batch_size, 12, num_samples]
pred = pred.transpose((1, 0, 2)).reshape((pred.shape[1], -1, num_samples)).transpose((1, 0, 2))
```

### Output Shape Transformation

```python
# Single sample
input_shape = [1, 96, 1]  # 1 sample, 96 timepoints, 1 feature

# After repeating for num_samples=10
repeated_shape = [10, 96, 1]

# After model forward pass
output_shape = [10, 12, 1]  # 10 predictions, 12 future points each

# After reshaping
final_shape = [1, 12, 10]  # 1 sample, 12 timepoints, 10 stochastic samples
```

### Uncertainty Visualization

**From predictions shape [n_samples, 12, 10]:**

For each of 12 time points:
- 10 different predicted values (from 10 forward passes)
- Compute KDE (kernel density estimation) to get probability distribution
- Plot as gradient-colored probability fan

**Key Statistics:**
- **Median**: Most likely prediction (robust central tendency)
- **Std Dev**: Model's uncertainty magnitude
- **Percentiles**: Confidence intervals (e.g., 5th-95th for 90% CI)

### num_samples Trade-offs

| Value | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| 1 | Fastest | No uncertainty | Quick demo |
| 10 | Fast | Good uncertainty | Web app (current) |
| 100 | Slow | Better uncertainty | Research/Training |
| 1000 | Very slow | Best uncertainty | Publication-grade |

---

## Model Prediction Flow

### Gluformer.predict() Detailed

**File:** `gluformer/model.py`

```python
def predict(self, test_dataset, batch_size=32, num_samples=100, device='cuda'):
    predictions = []
    logvars = []
    
    for batch in test_loader:
        past_target, historic_covs, future_covs, static_covs = batch
        # Shape: [batch_size * num_samples, 96, features]
        
        # Create decoder input (teacher forcing setup, but zeros for future)
        dec_inp = torch.cat([
            past_target[:, -self.label_len:, :],  # Last 32 points (label_len = 96/3)
            torch.zeros([past_target.shape[0], self.len_pred, 1])  # 12 zeros
        ], dim=1)  # Shape: [batch * num_samples, 44, 1]
        
        # Forward pass (dropout active due to train() mode)
        pred, logvar = self(static_covs, past_target, historic_covs, dec_inp, future_covs)
        # pred shape: [batch * num_samples, 12, 1]
        
        # Reshape to separate stochastic samples
        pred = pred.transpose((1, 0, 2)) \
                   .reshape((pred.shape[1], -1, num_samples)) \
                   .transpose((1, 0, 2))
        # Final: [batch_size, 12, num_samples]
        
        predictions.append(pred)
        logvars.append(logvar)
    
    return np.concatenate(predictions, axis=0), np.concatenate(logvars, axis=0)
```

### Architecture Summary

```
Input Components:
├─ Static Covariates: [batch, 1] (Subject ID)
├─ Past Target: [batch, 96, 1] (Glucose history)
├─ Historic Future Covariates: [batch, 96, 6] (Datetime features for input period)
└─ Future Covariates: [batch, 12, 6] (Datetime features for prediction period)

Encoder:
├─ Data Embedding (combines static + target + historic covariates)
├─ 2 Encoder Layers
│  ├─ Multi-head self-attention
│  ├─ Feed-forward network
│  └─ Dropout (r_drop=0.2)
└─ Output: [batch, 96, d_model=512]

Decoder Input:
└─ [Last 32 points of input | 12 zeros] → [batch, 44, 1]

Decoder:
├─ Data Embedding
├─ 2 Decoder Layers
│  ├─ Masked self-attention
│  ├─ Cross-attention to encoder output
│  ├─ Feed-forward network
│  └─ Dropout (r_drop=0.2)
└─ Output: [batch, 44, d_model=512]

Output Projection:
├─ Linear layer: d_model → 1
├─ Take last 12 positions (prediction window)
├─ Pred: [batch, 12, 1]
└─ LogVar: [batch, 12, 1] (for heteroscedastic uncertainty)
```

---

## Plotting & Visualization

### plot_forecast() (tools.py)

**Steps:**
1. **Select sample**: `forecasts[ind, :, :]` → [12, 10]
2. **Inverse transform**: Unscale using `scalers['target']`
3. **Get ground truth**: `dataset.evalsample(ind)` → actual glucose values at [pos+96:pos+108]
4. **Get input context**: `dataset[ind][0]` → past 96 points
5. **Generate uncertainty visualization**:
   - Add Gaussian noise to 10 samples for smoothing
   - For each of 12 time points: compute KDE from samples
   - Plot probability distribution as gradient-colored fan
6. **Plot elements**:
   - Blue line: True glucose (past 12 + future 12)
   - Red line: Median of 10 predictions
   - Gradient fans: Probability distribution at each future point

---

## Configuration Changes

### Old: config.yaml (Training-Centric)

```yaml
data_formatter_params:
  scaler_params: None
  use_time_features: True
  time_features: ['day', 'month', 'year', 'hour', 'minute', 'second']
  observation_interval: 5  # minutes
  use_rolling_statistics: False
  
split_params:
  length_segment: 13
  random_state: 0
  test_percent_subjects: 0.1

max_length_input: 192
length_pred: 12
```

### New: GluformerInferenceConfig (Inference-Optimized)

```python
class GluformerInferenceConfig(BaseModel):
    input_chunk_length: int = 96
    output_chunk_length: int = 12
    datetime_features: List[str] = ['day', 'month', 'year', 'hour', 'minute', 'second']
    observation_interval_minutes: int = 5
    gap_threshold_minutes: int = 45
    use_static_covariates: bool = True
    array_output_only: bool = True
```

**Key Differences:**
- Removed split parameters (no splitting in inference)
- Pydantic validation for type safety
- Explicit defaults for all parameters
- No training-specific parameters

---

## Comparative Summary

| Aspect | Old Pipeline | New Pipeline |
|--------|--------------|--------------|
| **Data Utilization** | 12% (Test split) | 100% (All data) |
| **Entry Point** | `utils/darts_processing.py::load_data()` | `fast_inference.py::create_inference_dataset_fast()` |
| **Data Processing** | `DataFormatter` (training class) | `FormatProcessor` (inference-optimized) |
| **Interpolation** | `formatter_utils.interpolate()` | Same + time-weighted in FormatProcessor |
| **Splitting** | Mandatory 4-way split | No splitting |
| **Scaler Fitting** | Train split (discarded data) | Global (actual user data) |
| **Config Source** | `config.yaml` | `GluformerInferenceConfig` (Pydantic) |
| **Quality Feedback** | None | Flags (IMPUTATION, CALIBRATION, duration) |
| **Inference Samples** | ~136 (from 3287 points) | ~1000+ (from 3287 points) |
| **Mathematical Accuracy** | Baseline | Identical (1e-16 precision) |

---

## Key Code References

### Old Pipeline
- **Entry**: `tools.py::prep_predict_glucose_tool()` → `utils/darts_processing.py::load_data()`
- **Formatter**: `data_formatter/base.py::DataFormatter.__init__()`
- **Splitting**: `data_formatter/utils.py::split()` (lines 172-300)
- **Scaling**: `utils/darts_processing.py::ScalerCustom`
- **Dataset**: `utils/darts_dataset.py::SamplingDatasetInferenceDual`

### New Pipeline
- **Entry**: `tools.py::prep_predict_glucose_tool()` → `fast_inference.py::create_inference_dataset_fast()`
- **Processor**: `cgm_format/src/cgm_format/format_processor.py::FormatProcessor`
- **Config**: `fast_inference.py::GluformerInferenceConfig` (Pydantic model)
- **Compatibility Functions**: Reuses `formatter_utils.interpolate()` and `formatter_utils.encode()`
- **Dataset**: Same `utils/darts_dataset.py::SamplingDatasetInferenceDual`

### Model Inference (Unchanged)
- **Prediction**: `gluformer/model.py::Gluformer.predict()`
- **Collation**: `gluformer/utils/training.py::modify_collate()`
- **Plotting**: `tools.py::plot_forecast()`

---

## Verification Results (proof_pipeline.py)

**Test Setup:**
- Input: Same raw data (previous.csv, 3287 points)
- Old pipeline: Process through DataFormatter → extract Test split
- New pipeline: Feed Test split equivalent (with 1 context point)

**Measurements:**
```python
# Shape verification
assert old_tensor.shape == new_tensor.shape  # ✅ [410, 96, 1]

# Value verification
max_abs_diff = np.max(np.abs(old_tensor - new_tensor))
assert max_abs_diff < 1e-15  # ✅ 1e-16 (floating point equality)

# Statistical verification
assert np.allclose(old_tensor, new_tensor, atol=1e-15, rtol=0)  # ✅ Pass
```

**Conclusion:** The refactor maintains perfect mathematical fidelity while improving efficiency and data utilization.

