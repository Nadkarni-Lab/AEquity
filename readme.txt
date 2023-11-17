Instrutions for use:

1. Install requirements: pip install -r requirements.txt

2. Set up config.yaml entries:
data_path: ./data/custom_data.tsv # Contains path to data. Contains independent variables, demographics, and outcome variables. 
demographics_col: demographics # Name of demographics variable in data_path.
outcome_cols: outcome_1  # Name of outcome variable in data_path
exclude_cols: None # name of columns to exclude if there are extraneous columns. 
out_data: ./output/data.tsv # Output directory for AEq analyses. 
bootstraps: 10 # Number of bootstraps. 30-50 is typically recommended for resolution at 5000 samples. 
start_seed: 42 # Seed experiments. 
input_dim: 149 # Number of independent columns in data_path
max_sample_size: 5000 # Max sample size to calculate from. Usually only require 128-512 samples. 
root_dir: ./weights # Root directory to output weights, and other output files. 

3. Run the measure and mitigate experiments:
python measure_disparity.py --config config.yaml
python mitigate_disparity.py --config config.yaml

Additional configurations are built for more complex models (AlexNet, ResNet, EfficientNet), but require a custom dataloader.
Custom  dataloders can be built by modifying the cnnMCSE repository. See the cnnMCSE repository for more details.
