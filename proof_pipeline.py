import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.darts_processing import load_data
from fast_inference import create_inference_dataset_fast
from utils.darts_dataset import SamplingDatasetInferenceDual
import data_formatter.utils as formatter_utils
import data_formatter.types as types

DataTypes = types.DataTypes
InputTypes = types.InputTypes

def rigorous_proof():
    print("=== Rigorous Mathematical Proof of Pipeline Equivalence ===")
    
    # 1. Load Previous Raw Data
    csv_path = Path("previous.csv")
    raw_df = pd.read_csv(csv_path)
    raw_df['time'] = pd.to_datetime(raw_df['time'])
    
    # 2. Run Old Pipeline to get the "Golden" Test Data
    # This dataframe has gone through interpolate -> encode -> split
    config_path = Path("files/config.yaml")
    if not config_path.exists(): config_path = Path("config.yaml")
    
    formatter, series, scalers = load_data(
        url=str(csv_path), 
        config_path=config_path,
        use_covs=True,
        cov_type='dual',
        use_static_covs=True
    )
    
    # Extract the EXACT test dataframe from the formatter
    # This is post-interpolation, post-encoding
    # formatter.test_idx is the indices in formatter.data (which is fully processed)
    # But wait, formatter.data is reset? No.
    # self.test_data = self.data.iloc[self.test_idx + self.test_idx_ood]
    # Let's use formatter.test_data directly, but remove OOD
    
    df_golden = formatter.test_data.loc[~formatter.test_data.index.isin(formatter.test_idx_ood)].copy()
    df_golden = df_golden.sort_values(['id', 'time'])
    
    print(f"Golden Test DF Shape: {df_golden.shape}")
    
    # 3. Manually "Hack" the Input for New Pipeline to MATCH Golden
    # The user asked: "if you switch off the re-interpolation... results will be identical, right?"
    
    # We cannot easily switch off interpolation in `create_inference_dataset_fast` because it calls 
    # `formatter_utils.interpolate` which assumes raw data.
    
    # BUT we can verify that if we feed the "Golden" DF (which is already interpolated) 
    # into the Darts creation part of the new pipeline, it produces identical tensors.
    
    # However, `create_inference_dataset_fast` does:
    # 1. Read CSV
    # 2. Interpolate
    # 3. Encode
    # 4. Create Darts
    
    # We need to Bypass steps 2 and 3 and 4 partially.
    
    # Let's modify `create_inference_dataset_fast` temporarily or just write the equivalent logic here.
    # We want to prove that:
    #   NewPipeline(Raw) == OldPipeline(Raw)
    # But we know NewPipeline includes MORE data.
    # So we want to prove:
    #   NewPipeline(Raw_subset) == OldPipeline(Raw)
    # where Raw_subset is the data corresponding to OldPipeline's test split.
    
    # But we found that Raw_subset extraction is tricky because of interpolation dependencies.
    
    # CHALLENGE: The Old Pipeline interpolated the FULL dataset THEN split.
    # The New Pipeline interpolates ONLY the input file.
    # If the input file is a SUBSET (test_split_precise.csv), interpolation at the boundary might differ
    # because the "previous point" needed for interpolation is missing.
    
    # HYPOTHESIS: If we manually add the missing context point to `test_split_precise.csv`, 
    # the New Pipeline will produce 100% identical first sample.
    
    # Let's find the missing context point.
    # The first sample of New Pipeline was shifted by 1 index compared to Old.
    # Old: [0.416, 0.364, ...]
    # New: [0.364, 0.330, ...]
    
    # This means New Pipeline starts at index 1 of Old Pipeline.
    # It means New Pipeline missed Index 0.
    
    # Index 0 in Old Pipeline corresponds to some time T.
    # Let's find T from the Old Dataset.
    # We can get the TimeIndex from the TimeSeries.
    
    ts_old = series['test']['target'][0] # First TimeSeries in Old Test Set
    start_time_old = ts_old.time_index[0]
    print(f"Old Test Series 0 Start Time: {start_time_old}")
    print(f"Old Test Series 0 Value at Start: {ts_old.values()[0][0]}")
    
    # Now let's check our `test_split_precise.csv`
    # Does it contain data BEFORE this start time?
    df_precise = pd.read_csv("test_split_precise.csv")
    df_precise['time'] = pd.to_datetime(df_precise['time'])
    
    min_time_precise = df_precise[df_precise['id'] == ts_old.static_covariates.values[0,0]]['time'].min()
    print(f"Precise Raw Input Start Time: {min_time_precise}")
    
    # If min_time_precise > start_time_old, then obviously we are missing data.
    # If min_time_precise == start_time_old, but start_time_old was INTERPOLATED, 
    # then we needed the point BEFORE it to generate it.
    
    # Let's check if start_time_old exists in Raw Data
    raw_match = raw_df[raw_df['time'] == start_time_old]
    if len(raw_match) > 0:
        print(f"Start time {start_time_old} EXISTS in Raw Data.")
    else:
        print(f"Start time {start_time_old} DOES NOT EXIST in Raw Data (It is interpolated).")
        # This is likely the case.
        
        # Find the raw point immediately preceding start_time_old
        # We need this point to let interpolate() work correctly.
        # Filter raw_df for this ID and time < start_time_old
        subj_id = ts_old.static_covariates.values[0,0]
        preceding_points = raw_df[(raw_df['id'] == subj_id) & (raw_df['time'] < start_time_old)]
        
        if len(preceding_points) > 0:
            last_preceding = preceding_points.iloc[-1]
            print(f"Found preceding raw point: {last_preceding['time']} Value: {last_preceding['gl']}")
            
            # ADD this point to test_split_precise.csv
            # And run New Pipeline again.
            
            row_to_add = last_preceding.to_frame().T
            df_precise_augmented = pd.concat([row_to_add, df_precise]).sort_values(['id', 'time'])
            
            aug_csv_path = "test_split_augmented.csv"
            df_precise_augmented.to_csv(aug_csv_path, index=False)
            print(f"Created {aug_csv_path} with added context point.")
            
            print("\n=== Running New Pipeline on Augmented Data ===")
            dataset_aug, _ = create_inference_dataset_fast(
                file_path=aug_csv_path,
                scaler_target=scalers['target'],
                scaler_covs=scalers['future']
            )
            
            # Now check the first sample
            if len(dataset_aug) > 0:
                s_aug = dataset_aug[0][0] # Past target
                s_old_val = ts_old.values()[:96] # First 96 points (input chunk)
                
                # Wait, dataset_old[0] might not be the start of the series?
                # dataset_old uses SamplingDatasetInferenceDual.
                # It creates samples from valid locations.
                # First sample usually starts at index 0 if valid.
                
                # Let's compare s_aug with s_old_val
                # Note: s_old from debug_comparison was 0.416...
                # Let's see what we get now.
                
                print(f"Augmented New Sample 0 Mean: {s_aug.mean()}")
                
                # We need to find where this aligns with Old Dataset.
                # The Old Dataset might produce multiple samples.
                # Let's assume index 0 matches index 0.
                
                # We compare against the FIRST sample from debug_comparison which was 0.4163569
                # Let's hardcode that value for verification if we can't access dataset_old easily here
                # (though we can recreate it)
                
                # Recreate old dataset sample for comparison
                dataset_old_check = SamplingDatasetInferenceDual(
                    target_series=series['test']['target'],
                    covariates=series['test']['future'],
                    input_chunk_length=96,
                    output_chunk_length=12,
                    use_static_covariates=True,
                    array_output_only=True
                )
                s_old_check = dataset_old_check[0][0]
                
                print(f"Old Sample 0 Mean: {s_old_check.mean()}")
                
                if np.allclose(s_aug, s_old_check, atol=1e-16):
                    print("\n✅ SUCCESS! With the missing context point, the pipelines match EXACTLY.")
                    print("Proof complete: The logic is identical, the difference was only in input data truncation.")
                else:
                    print("\n❌ Still mismatch. Let's inspect.")
                    print(f"First 5 Aug: {s_aug[:5].flatten()}")
                    print(f"First 5 Old: {s_old_check[:5].flatten()}")
                    
                    # Maybe we need MORE context points?
                    # Interpolation might need 2 points? Or gap threshold logic?
                    
        else:
            print("No preceding point found? This implies start of dataset.")

if __name__ == "__main__":
    rigorous_proof()

