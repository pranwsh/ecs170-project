"""Inspect supplied image datasets and save training sample/count figures."""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Resolve project imports independently of the directory used to launch the script.
TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from local_code.stage_3_code.Dataset_Loader import DATA_DIRECTORY
from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
from local_code.stage_3_code.Visualize_Data import save_visualizations


def main():
    # Default to project data/result folders and allow explicit overrides.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIRECTORY)
    parser.add_argument('--output-dir', type=Path,
                        default=TEMPLATE_ROOT / 'result' / 'stage_3_result' / 'dataset_inspection')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Expected partition sizes and processed image shapes come from the dataset contract.
    specifications = ((Dataset_Loader_MNIST, 60000, 10000, (1, 28, 28)),
                      (Dataset_Loader_ORL, 360, 40, (1, 112, 92)),
                      (Dataset_Loader_CIFAR, 50000, 10000, (3, 32, 32)))
    summary = {}
    for loader_class, train_count, test_count, shape in specifications:
        # Hash the source file so recorded checks identify the exact input data.
        path = args.data_dir / loader_class.file_name
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
            source.seek(0)
            stored = pickle.load(source)

        # Record original values before the loaders normalize or rearrange images.
        details = {'sha256': digest, 'partitions': {}}
        original_labels = {}
        for split in ('train', 'test'):
            records = stored[split]
            original_labels[split] = np.asarray([record['label'] for record in records])
            details['partitions'][split] = {
                'stored_shape': list(np.asarray(records[0]['image']).shape),
                'stored_dtype': str(np.asarray(records[0]['image']).dtype),
                'pixel_min': float(min(np.min(record['image']) for record in records)),
                'pixel_max': float(max(np.max(record['image']) for record in records)),
                'original_label_min': int(original_labels[split].min()),
                'original_label_max': int(original_labels[split].max()),
            }
            del records
        # Release raw images before loading processed arrays to limit peak memory.
        del stored

        for split, expected_count in (('train', train_count), ('test', test_count)):
            loader = loader_class(split=split, data_dir=args.data_dir)
            data = loader.load()
            images, labels = data['X'], data['y']

            # Verify the complete partition's dimensions, types, range, and label order.
            assert images.shape == (expected_count, *shape)
            assert labels.shape == (expected_count,)
            assert images.dtype == np.float32 and labels.dtype == np.int64
            assert images.flags.c_contiguous and np.isfinite(images).all()
            assert 0 <= images.min() <= images.max() <= 1
            assert set(labels.tolist()) == set(range(loader.num_classes))
            np.testing.assert_array_equal(loader.display_labels(labels), original_labels[split])
            details['partitions'][split].update({
                'output_shape': list(images.shape),
                'output_dtype': str(images.dtype),
                'label_dtype': str(labels.dtype),
                'normalized_min': float(images.min()),
                'normalized_max': float(images.max()),
                'class_counts': np.bincount(labels, minlength=loader.num_classes).tolist(),
            })

            # Only supplied training images appear in sample grids and count charts.
            if split == 'train':
                save_visualizations(data, loader.file_name, args.output_dir,
                                    label_offset=loader.label_offset)
            del images, labels, data, loader

        summary[loader_class.file_name] = details
        print(f'{loader_class.file_name}: verified train={train_count}, test={test_count}', flush=True)

    # Store measured metadata alongside the six training-data figures.
    output_path = args.output_dir / 'dataset_checks.json'
    output_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Saved dataset checks and figures to {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
