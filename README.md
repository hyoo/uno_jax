# Jax/Flax implementation of UNO model
The original model was written in Keras, which can be found at https://github.com/ECP-CANDLE/Benchmarks/tree/master/Pilot1/Uno

```
$ python train.py -e 10 --lr 4e-3 -z 512
+--------------------------------+--------------+-----------+-----------+--------+
| Name                           | Shape        | Size      | Mean      | Std    |
+--------------------------------+--------------+-----------+-----------+--------+
| params/Dense_0/bias            | (1,)         | 1         | 0.0       | 0.0    |
| params/Dense_0/kernel          | (1000, 1)    | 1,000     | 0.000583  | 0.032  |
| params/dense_0/bias            | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/dense_0/kernel          | (2000, 1000) | 2,000,000 | 4.27e-05  | 0.0258 |
| params/dense_1/bias            | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/dense_1/kernel          | (1000, 1000) | 1,000,000 | 1.02e-05  | 0.0316 |
| params/dense_2/bias            | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/dense_2/kernel          | (1000, 1000) | 1,000,000 | 2.23e-06  | 0.0316 |
| params/dense_3/bias            | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/dense_3/kernel          | (1000, 1000) | 1,000,000 | -4.77e-05 | 0.0316 |
| params/dense_4/bias            | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/dense_4/kernel          | (1000, 1000) | 1,000,000 | -2.04e-06 | 0.0316 |
| params/drug_net/dense_0/bias   | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/drug_net/dense_0/kernel | (5270, 1000) | 5,270,000 | -1.2e-05  | 0.0179 |
| params/drug_net/dense_1/bias   | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/drug_net/dense_1/kernel | (1000, 1000) | 1,000,000 | -2.22e-05 | 0.0316 |
| params/drug_net/dense_2/bias   | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/drug_net/dense_2/kernel | (1000, 1000) | 1,000,000 | 2.89e-05  | 0.0316 |
| params/gene_net/dense_0/bias   | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/gene_net/dense_0/kernel | (942, 1000)  | 942,000   | -8.78e-05 | 0.0321 |
| params/gene_net/dense_1/bias   | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/gene_net/dense_1/kernel | (1000, 1000) | 1,000,000 | -4.11e-05 | 0.0316 |
| params/gene_net/dense_2/bias   | (1000,)      | 1,000     | 0.0       | 0.0    |
| params/gene_net/dense_2/kernel | (1000, 1000) | 1,000,000 | -3.4e-05  | 0.0316 |
+--------------------------------+--------------+-----------+-----------+--------+
Total: 16,224,001
epoch:   1, elapsed:15.2, train_loss: 0.1033, val_loss: 0.0896
epoch:   2, elapsed:11.0, train_loss: 0.0882, val_loss: 0.0853
epoch:   3, elapsed:12.7, train_loss: 0.0849, val_loss: 0.0829
epoch:   4, elapsed:11.8, train_loss: 0.0826, val_loss: 0.0809
epoch:   5, elapsed:11.5, train_loss: 0.0809, val_loss: 0.0793
epoch:   6, elapsed:11.2, train_loss: 0.0793, val_loss: 0.0781
epoch:   7, elapsed:11.6, train_loss: 0.0777, val_loss: 0.0774
epoch:   8, elapsed:11.6, train_loss: 0.0765, val_loss: 0.0764
epoch:   9, elapsed:10.8, train_loss: 0.0752, val_loss: 0.0769
epoch:  10, elapsed:11.2, train_loss: 0.0741, val_loss: 0.0754
```