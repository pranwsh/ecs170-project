"""Run from project template root: python -m script.stage_3_script.run_experiment --help."""
import argparse
from local_code.stage_3_code.experiments import DATASETS, VARIANTS, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--mode', choices=('validation', 'test'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='baseline')
    parser.add_argument('--output', required=True, help='New output directory; existing directories are not overwritten')
    parser.add_argument('--data-dir')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--split-seed', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(run_experiment(args.dataset, args.mode, args.variant, data_dir=args.data_dir,
                         output=args.output, seed=args.seed, split_seed=args.split_seed,
                         epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.learning_rate, device=args.device))


if __name__ == '__main__':
    main()
