import argparse
import yaml
from cnnMCSE.predict import measure_disparity_custom


def main(**config_kwargs):
    measure_disparity_custom(**(config_kwargs))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for convergence sample size estimtation.')
    parser.add_argument("--config", required=True, type=str,
                        help="Config file path.")
    config_kwargs = parser.parse_args()

    opt = vars(config_kwargs)
    config_kwargs = yaml.load(open(config_kwargs.config), Loader=yaml.FullLoader)
    opt.update(config_kwargs)
    config_kwargs = opt


    main(**(config_kwargs))