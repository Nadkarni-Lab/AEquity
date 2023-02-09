import itertools
import argparse
import yaml
import numpy as np
import pandas as pd
from cnnMCSE.predict import mitigate_disparity_custom



class MitigateDisparity():
    def __init__(self, config_kwargs):
        df, preds_df = mitigate_disparity_custom(**(config_kwargs))
        self.df = df
        self.preds_df = preds_df
    
    def fit(self):
        return self.df
    
    def predict(self):
        return self.preds_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for convergence sample size estimtation.')
    parser.add_argument("--config", required=True, type=str,
                        help="Config file path.")
    config_kwargs = parser.parse_args()

    opt = vars(config_kwargs)
    config_kwargs = yaml.load(open(config_kwargs.config), Loader=yaml.FullLoader)
    opt.update(config_kwargs)
    config_kwargs = opt

    mitigate_disparity = MitigateDisparity((config_kwargs))
