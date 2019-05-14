from exp_recorder import Exp_Recorder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

recorder = Exp_Recorder([])

exp_list = ['2p_A', '2p_B', '2p_C',
            '3p_A', '3p_B', '7p']

step = 100
settings = ['2 parties – A', '2 parties – B', '2 parties – C',
            '3 parties – A', '3 parties – B', '7 parties']
d = {'Iteration': [], 'Setting': [], 'Accuracy': []}
for idx, e in enumerate(exp_list):
    setting = settings[idx]
    for i in range(10):
        recorder.load(os.path.join('..', 'exp', '%s_%d.pkl' % (e, i + 1)))
        acc_list = recorder.record['global'][1:]
        if len(acc_list) < step:  # converge before using up budget
            acc_list.extend([acc_list[-1]] * (step - len(acc_list)))
        if len(acc_list) > step:
            acc_list = acc_list[:step]
        d['Setting'].extend([setting] * step)
        d['Iteration'].extend(list(range(1, step + 1)))
        d['Accuracy'].extend(acc_list)

df = pd.DataFrame(data=d)

sns.set()
sns.set_style("ticks")
sns.set_context(rc={"lines.linewidth": 2.5})
my_palette = ['#E23E3D', '#6998C5', '#78C072', '#A96FB4', '#F99833', '#FFCC00']
mycmap = sns.color_palette(my_palette)
b = sns.lineplot(x='Iteration', y='Accuracy', hue='Setting', data=df, palette=mycmap, ci='sd')
b.set_xlabel("Iteration", fontsize=15)
b.set_ylabel("Accuracy", fontsize=15)
b.tick_params(labelsize=13)
plt.setp(b.get_legend().get_texts(), fontsize='13')
sns.despine()
plt.savefig(os.path.join('..', 'figure.pdf'), bbox_inches='tight')
plt.show()
