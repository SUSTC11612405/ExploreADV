import pandas as pd
import matplotlib.pyplot as plt

topk = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50]


def plot_line(name):
    file_path1 = name + '.csv'
    file_path = name+'_topk.csv'
    df_raw = pd.read_csv(file_path1)
    df = pd.read_csv(file_path)

    df2 = df_raw.merge(df, on='index')
    for k in topk:
        df2['r_{}'.format(k)] = df2['top{}'.format(k)] / df2['min l_inf']
    print(df2.mean())

    value = [df2['r_{}'.format(k)].mean() for k in topk]
    plt.plot(topk, value, label=label[name])


names = ['mnist_relu_9_200', 'mnist_convSmallRELU', 'cifar10_relu_6_500',
         'cifar10_convMedGSIGMOID', 'cifar10_convBigRELU', 'cifar10_ResNet18']
label = {'mnist_relu_9_200':'M1', 'mnist_convSmallRELU':'M2', 'cifar10_relu_6_500':'C1',
         'cifar10_convMedGSIGMOID':'C2', 'cifar10_convBigRELU':'C3', 'cifar10_ResNet18':'C4'}
# fig, ax = plt.subplots(1, 1)
# for name in names:
#     plot_line(name)
# plt.legend()
# plt.ylabel('ratio')
# plt.xlabel('k')
# ax.set_xticks([1, 3, 5, 7, 10, 15, 20, 30, 40, 50])
# plt.show()

df = pd.read_csv('../results/topk_timecost.csv')
fig, ax = plt.subplots(1, 1)
for _, row in df.iterrows():
    value = [row['top{}'.format(k)] for k in topk]
    plt.plot(topk, value, label=row['model'])
plt.legend()
plt.ylabel('time (seconds)')
plt.xlabel('k')
ax.set_xticks([1, 3, 5, 7, 10, 15, 20, 30, 40, 50])
plt.show()
