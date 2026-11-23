# Stage 3 experiment summary

Metrics are fractions (0–1); precision, recall and F1 are macro-averaged.
Validation is for model selection; test results are final evaluations and must not be used to select variants.

## Validation ablations

| Dataset | Variant | Run | Accuracy | Macro precision | Macro recall | Macro F1 | Correct / total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cifar | average_pool | cifar/average_pool_20epoch_validation_seed2 | 0.7156 | 0.7172 | 0.7156 | 0.7133 | 3578/5000 |
| cifar | average_pool | cifar/average_pool_2epoch | 0.5092 | 0.5207 | 0.5092 | 0.5047 | 2546/5000 |
| cifar | average_pool | cifar/average_pool_4epoch_validation_seed2 | 0.5684 | 0.5748 | 0.5684 | 0.5577 | 2842/5000 |
| cifar | baseline | cifar/baseline_20epoch_validation_seed2 | 0.7258 | 0.7347 | 0.7258 | 0.7275 | 3629/5000 |
| cifar | baseline | cifar/baseline_2epoch | 0.5748 | 0.5968 | 0.5748 | 0.5758 | 2874/5000 |
| cifar | baseline | cifar/baseline_4epoch_validation_seed2 | 0.6568 | 0.6604 | 0.6568 | 0.6520 | 3284/5000 |
| cifar | kernel5 | cifar/kernel5_20epoch_validation_seed2 | 0.7290 | 0.7355 | 0.7290 | 0.7300 | 3645/5000 |
| cifar | kernel5 | cifar/kernel5_2epoch | 0.6198 | 0.6311 | 0.6198 | 0.6215 | 3099/5000 |
| cifar | kernel5 | cifar/kernel5_4epoch_validation_seed2 | 0.6836 | 0.6964 | 0.6836 | 0.6801 | 3418/5000 |
| mnist | average_pool | mnist/average_pool | 0.9862 | 0.9861 | 0.9860 | 0.9860 | 5917/6000 |
| mnist | baseline | mnist/baseline | 0.9868 | 0.9868 | 0.9866 | 0.9867 | 5921/6000 |
| mnist | kernel5 | mnist/kernel5 | 0.9895 | 0.9895 | 0.9894 | 0.9894 | 5937/6000 |
| orl | average_pool | orl/average_pool | 0.9250 | 0.8875 | 0.9250 | 0.9000 | 37/40 |
| orl | baseline | orl/baseline | 0.9250 | 0.8875 | 0.9250 | 0.9000 | 37/40 |
| orl | kernel5 | orl/kernel5 | 0.9250 | 0.8875 | 0.9250 | 0.9000 | 37/40 |
| orl | average_pool | orl/repeat_seed3_average_pool | 0.9750 | 0.9625 | 0.9750 | 0.9667 | 39/40 |
| orl | baseline | orl/repeat_seed3_baseline | 0.9500 | 0.9250 | 0.9500 | 0.9333 | 38/40 |
| orl | kernel5 | orl/repeat_seed3_kernel5 | 0.9250 | 0.8875 | 0.9250 | 0.9000 | 37/40 |
| orl | average_pool | orl/repeat_seed4_average_pool | 0.9250 | 0.9000 | 0.9250 | 0.9083 | 37/40 |
| orl | baseline | orl/repeat_seed4_baseline | 0.9250 | 0.8875 | 0.9250 | 0.9000 | 37/40 |
| orl | kernel5 | orl/repeat_seed4_kernel5 | 0.9750 | 0.9625 | 0.9750 | 0.9667 | 39/40 |

## Final test evaluations

| Dataset | Variant | Run | Accuracy | Macro precision | Macro recall | Macro F1 | Correct / total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cifar | kernel5 | cifar/final_2epoch | 0.6120 | 0.6216 | 0.6120 | 0.6022 | 6120/10000 |
| mnist | kernel5 | mnist/final | 0.9912 | 0.9913 | 0.9910 | 0.9911 | 9912/10000 |
| orl | baseline | orl/final | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40/40 |

## Run settings

| Run | Dataset | Split | Variant | Model options | Seed | Split seed | Epochs | Batch size | Learning rate | Device | Train count | Evaluation count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar/average_pool_20epoch_validation_seed2 | cifar | validation | average_pool | {"pooling": "average"} | 2 | 2 | 20 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/average_pool_2epoch | cifar | validation | average_pool | {"pooling": "average"} | 2 | 2 | 2 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/average_pool_4epoch_validation_seed2 | cifar | validation | average_pool | {"pooling": "average"} | 2 | 2 | 4 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/baseline_20epoch_validation_seed2 | cifar | validation | baseline | {} | 2 | 2 | 20 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/baseline_2epoch | cifar | validation | baseline | {} | 2 | 2 | 2 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/baseline_4epoch_validation_seed2 | cifar | validation | baseline | {} | 2 | 2 | 4 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/final_2epoch | cifar | test | kernel5 | {"kernel_size": 5} | 2 | 2 | 2 | 128 | 0.001 | cpu | 50000 | 10000 |
| cifar/kernel5_20epoch_validation_seed2 | cifar | validation | kernel5 | {"kernel_size": 5} | 2 | 2 | 20 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/kernel5_2epoch | cifar | validation | kernel5 | {"kernel_size": 5} | 2 | 2 | 2 | 128 | 0.001 | cpu | 45000 | 5000 |
| cifar/kernel5_4epoch_validation_seed2 | cifar | validation | kernel5 | {"kernel_size": 5} | 2 | 2 | 4 | 128 | 0.001 | cpu | 45000 | 5000 |
| mnist/average_pool | mnist | validation | average_pool | {"pooling": "average"} | 2 | 2 | 20 | 128 | 0.001 | cpu | 54000 | 6000 |
| mnist/baseline | mnist | validation | baseline | {} | 2 | 2 | 20 | 128 | 0.001 | cpu | 54000 | 6000 |
| mnist/final | mnist | test | kernel5 | {"kernel_size": 5} | 2 | 2 | 20 | 128 | 0.001 | cpu | 60000 | 10000 |
| mnist/kernel5 | mnist | validation | kernel5 | {"kernel_size": 5} | 2 | 2 | 20 | 128 | 0.001 | cpu | 54000 | 6000 |
| orl/average_pool | orl | validation | average_pool | {"pooling": "average"} | 2 | 2 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/baseline | orl | validation | baseline | {} | 2 | 2 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/final | orl | test | baseline | {} | 2 | 2 | 20 | 32 | 0.001 | cpu | 360 | 40 |
| orl/kernel5 | orl | validation | kernel5 | {"kernel_size": 5} | 2 | 2 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/repeat_seed3_average_pool | orl | validation | average_pool | {"pooling": "average"} | 3 | 3 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/repeat_seed3_baseline | orl | validation | baseline | {} | 3 | 3 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/repeat_seed3_kernel5 | orl | validation | kernel5 | {"kernel_size": 5} | 3 | 3 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/repeat_seed4_average_pool | orl | validation | average_pool | {"pooling": "average"} | 4 | 4 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/repeat_seed4_baseline | orl | validation | baseline | {} | 4 | 4 | 20 | 32 | 0.001 | cpu | 320 | 40 |
| orl/repeat_seed4_kernel5 | orl | validation | kernel5 | {"kernel_size": 5} | 4 | 4 | 20 | 32 | 0.001 | cpu | 320 | 40 |
