import pandas as pd

# file_path1 = 'mnist_convSmallRELU.csv'
# file_path = 'mnist_convSmallRELU_masks.csv'
# file_path1 = 'cifar10_convMedGSIGMOID.csv'
# file_path = 'cifar10_convMedGSIGMOID_masks.csv'
file_path1 = 'cifar10_convBigRELU.csv'
file_path = 'cifar10_convBigRELU_masks.csv'
df_raw = pd.read_csv(file_path1)
df = pd.read_csv(file_path)

print(df.mean())
# print(df['captum'].value_counts()[1.0])
# print(df['captum_correction'].value_counts()[1.0])
# print(df['gradcam'].value_counts()[1.0])
# print(df['gradcam++'].value_counts()[1.0])

df2 = df_raw.merge(df, on='index')
print(df2)

df2['r_captum'] = df2['captum'] / df2['min l_inf']
df2['r_captum_correction'] = df2['captum_correction'] / df2['min l_inf']
df2['r_gradcam'] = df2['gradcam'] / df2['min l_inf']
df2['r_gradcam++'] = df2['gradcam'] / df2['min l_inf']
print(df2.mean())
