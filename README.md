# CEXCustomized
| sample | delta = epsilon | delta = 0 | 0 < delta < epsilon |
| ---- | ---- | ---- | ---- |
|MNIST sample 1|480 |269 |35|
|MNIST sample 2|448 |296 |40|
|MNIST sample 3|453 |321 |10|
|MNIST sample 4|486 |249 |49|
|MNIST sample 5|426 |337 |21|
|CIFAR10 sample 1|3062 |0 |10|
|CIFAR10 sample 2|2875 |0 |40|
|CIFAR10 sample 3|3013 |4 |55|


### MNIST
| method | mnist_relu_9_200 | mnist_relu_6_100 | convSmallRELU__Point |
| ---- | ---- | ---- | ---- |
| DEEPFOOL | 0.07423 | 0.07331 | 0.1837 |
| FAB | 0.06379 | 0.06470 | 0.1571 |
| BB | 0.06358 | 0.06535 | 0.1558 |
| Ours | 0.06208 | 0.06281 | 0.1541 |
| Ours-I (50% pixels) | 0.08699 | 0.09044 | 0.1623 |

#### mnist_relu_9_200
![mnist_relu_9_200](images/mnist_relu_9_200.png)

### Cifar10
| method | ffnnRELU__Point_6_500 | convSmallRELU__Point | ResNet18_PGD |
| ---- | ---- | ---- | ---- |
| DEEPFOOL | 0.01328 | 0.008010 | 0.04297 |
| FAB | 0.008917 | 0.007489 | 0.03434, 0.03458(alpha_max=0) |
| BB | 0.009386 | 0.007895 | 0.03629 |
| Ours | 0.008189 | 0.007383 | 0.03451 |
| Ours-I (50% pixels) | 0.01657 | 0.01210 | 0.05569 |
| Ours-V (50% pixels) | 0.01817 | 0.01390 | 0.05534 |
| Ours-C (50% pixels) | 0.01677 | 0.01181 | 0.05100 |

#### cifar10_2_255
![cifar10_2_255](images/cifar10_2_255.png)
#### ResNet18_PGD
![ResNet18_PGD](images/cifar10_ResNet18_PGD.png)


### STL10
![stl10](images/stl10.png)
