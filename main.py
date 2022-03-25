import argparse
import os
from typing import Dict

import yaml
import torch
from torch.nn import DataParallel

from ddpm import Unet, GaussianDiffusion, Trainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(cfg: Dict) -> None:
    model = Unet(**cfg['MODEL'])
    model = DataParallel(model)

    diffusion = GaussianDiffusion(
        model,
        **cfg['DIFFUSION'],
        device=DEVICE,
    )

    trainer = Trainer(
        diffusion,
        **cfg['TRAINER'],
    )

    trainer.train()


def read_yml(filepath: str) -> dict:
    """Load a yml file to memory as dict."""
    with open(filepath, 'r') as ymlfile:
        return dict(yaml.load(ymlfile, Loader=yaml.FullLoader))


def main() -> None:
    """Execute main func."""
    # Get correct config file over command line.
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/01_config.yml', type=str)
    parser.add_argument('-s', '--stint', default=0, type=int)
    args = parser.parse_args()

    # Read yml file to memory as dict.
    cfg = read_yml(args.config)

    # Check if training is part of a multi-stint run
    if args.stint != 0:
        if args.stint == 1:
            cfg['TRAINER']['start_from_checkpoint'] = False
        else:
            cfg['TRAINER']['start_from_checkpoint'] = True
            cfg['TRAINER']['checkpoint_path'] = os.path.join(
                cfg['TRAINER']['results_folder'],
                'v' + str(args.stint - 1),
                'model.pt'
            )

        os.makedirs(cfg['TRAINER']['results_folder'], exist_ok=True)

        cfg['TRAINER']['results_folder'] = os.path.join(
            cfg['TRAINER']['results_folder'],
            'v' + str(args.stint)
        )
    # Start training with chosen configuration.
    run(cfg)


if __name__ == '__main__':
    main()
