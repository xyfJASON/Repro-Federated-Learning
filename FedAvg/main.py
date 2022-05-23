import argparse
from trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config.yml', help='path to training configuration file')
    args = parser.parse_args()

    trainer = Trainer(args.config_path)
    trainer.train()
