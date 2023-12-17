import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os


sys_names = ['LiHecp','BeHecp','COecp','H2Oecp','H2O','H2Secp']
result_names = ["LiH_ecp", "BeH_ecp", "CO_ecp", "H2O_ecp", "H2O_all", "H2S_ecp"]

for result_name in result_names:
    if not os.path.exists(result_name):
        os.makedirs(result_name)

ckpt_restore_filenames = []
for sys_name in sys_names:
    ckpt_restore_filename = []
    files = os.listdir('../data/expriment_2_1/' + sys_name)
    for file in files:
        file_name = '../data/expriment_2_1/'+sys_name+'/'+file+'/train_stats.csv'
        if 'ferminet_2023' in file and os.path.exists(file_name):
            ckpt_restore_filename.append(file_name)
    ckpt_restore_filename.sort()
    ckpt_restore_filenames.append(ckpt_restore_filename)

E_g = {}
E_g_var = {}
E_g_std = {}
E_exciteds = {}
E_excited_var = {}
E_excited_std = {}
T_out = {}
for sys in range(len(result_names)):
    print(ckpt_restore_filenames[sys])
    print('\n')
    E = []
    T = []
    Var = []
    S = []
    S_numpy = []
    Verli = []

    for file_name in ckpt_restore_filenames[sys]:
        df = pd.read_csv(file_name)
        S_index = []
        for col in df.columns:
            if "S_" in col:
                S_index.append(col)
        E.append(df['E_mean_clip'])
        T.append(df['delta_time'])
        Var.append(df['E_var_clip'])
        S.append(df[S_index])
        Verli.append((df['V_loc']+df['V_nloc']+df['V'])/df['T'])

    S_matrix = np.zeros((len(S), len(S), len(S[0])))
    for i in range(len(S)):
        for j in range(S[i].shape[1]):
            S_matrix[i][j] = S[i][S[i].columns[j]].values
    S_matrixs = S_matrix.swapaxes(0,1) + S_matrix+np.tile(np.eye(len(S))[..., None], len(S[0]))
    # 输出数据
    # 能量收敛原始数据
    Es = pd.DataFrame(E, index=[i for i in range(len(E))]).T
    Es.to_csv(result_names[sys]+'/E.csv', header=None, index=True)
    # 不确定度原始数据
    E_vars = pd.DataFrame(Var, index=[i for i in range(len(Var))]).T
    E_vars.to_csv(result_names[sys]+'/E_var.csv', header=None, index=True)
    # 重叠系数原始数据
    S = pd.concat(S, axis=1)
    S = pd.DataFrame(S, index=[i for i in range(S.shape[1])]).T
    S.to_csv(result_names[sys]+'/S.csv', header=None, index=True)
    # 位力定理原始数据
    Verlis = pd.DataFrame(Verli, index=[i for i in range(len(Verli))]).T
    Verlis.to_csv(result_names[sys]+'/Verli.csv', header=None, index=True)
    # 时间统计原始数据
    T = pd.DataFrame(T, index=[i for i in range(len(T))]).T
    T.to_csv(result_names[sys]+'/T.csv', header=None, index=True)
    # 能量重采样数据
    Es1 = Es.rolling(window=100, center=True, min_periods=1).mean()
    E_baseline = Es1.loc[Es1.index[-50]].min()
    Es1 -= E_baseline
    Es1[50::100].to_csv(result_names[sys]+'/E1.csv', header=None, index=True)
    E_std = Es.rolling(window=100, center=True, min_periods=1).std()
    E_std[50::100].to_csv(result_names[sys]+'/E_std.csv', header=None, index=True)
    # 能量不确定度重采样数据
    E_vars1 = E_vars.rolling(window=100, center=True, min_periods=1).mean()
    E_vars1[50::100].to_csv(result_names[sys]+'/E_var1.csv', header=None, index=True)
    # 重叠系数重采样数据
    S1 = S.rolling(window=100, center=True, min_periods=1).mean()
    S1[50::100].to_csv(result_names[sys]+'/S1.csv', header=None, index=True)
    # 重叠矩阵数据
    S_matrix = S_matrixs[:,:,-100:].mean(axis=2)
    pd.DataFrame(S_matrix).to_csv(result_names[sys]+'/S_matrix.csv', header=None, index=False)
    # 设置图片大小
    plt.figure(figsize=(5, 4))
    cmap = mpl.cm.get_cmap('YlGn')
    plt.imshow(S_matrix, norm=mpl.colors.LogNorm(vmin=S_matrix.min(), vmax=S_matrix.max()), cmap=cmap, alpha=1)
    plt.colorbar()
    # 保存图片
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.savefig(result_names[sys]+'/overlap.svg', dpi=600, bbox_inches='tight')
    # 位力定理重采样数据
    Verlis1 = Verlis.rolling(window=100, center=True, min_periods=1).mean()
    Verlis1[50::100].to_csv(result_names[sys]+'/Verli1.csv', header=None, index=True)
    # 耗时统计
    T1 = T.mean()
    E_baseline
    E_excited = Es1.loc[Es1.index[-50]]

    E_g[result_names[sys]] = E_baseline
    E_g_var[result_names[sys]] = E_vars1[Es1.loc[Es1.index[-50]].argmin()][Es1.index[-50]]
    E_g_std[result_names[sys]] = E_std[Es1.loc[Es1.index[-50]].argmin()][Es1.index[-50]]
    E_exciteds[result_names[sys]] = E_excited
    E_excited_var[result_names[sys]] = E_vars1.loc[Es1.index[-50]]
    E_excited_std[result_names[sys]] = E_std.loc[Es1.index[-50]]
    T_out[result_names[sys]] = T.mean()

E_base = {"E":E_g, "E_var":E_g_var, "E_std":E_g_std}
pd.DataFrame(E_base).to_csv('E_base.csv', header=True, index=True)

E_excit = {
    "E":pd.concat(E_exciteds, axis=0), 
    "E_var":pd.concat(E_excited_var, axis=0), 
    "E_std":pd.concat(E_excited_std, axis=0), 
    "T":pd.concat(T_out, axis=0)}

pd.DataFrame(E_excit).to_csv('E_excit.csv', header=True, index=True)