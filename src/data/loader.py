import pandas as pd
import glob
import os
from pathlib import Path
def load_skab(base_path="data/raw/SKAB"):
    """SKAB veri setini valve1 ve valve2 klasörlerinden okuyup birleştirir."""
    all_data = []
    
    for folder in ['valve1', 'valve2']:
        folder_path = Path(base_path) / folder
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file in csv_files:
            df = pd.read_csv(file, sep=';') # SKAB genelde noktalı virgül kullanır
            df['source_group'] = folder
            df['source_file'] = os.path.basename(file)
            all_data.append(df)
            
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df
