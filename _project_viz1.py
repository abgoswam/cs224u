import pandas as pd
import matplotlib.pyplot as plt

Data_ANLI_r1 = {
    'Percentage_Pruned': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Accuracy': [26, 25.5, 23.5, 24.6, 25, 24, 25.7, 26, 27.7, 31.5, 33.0]
}

# values in 60, 70, 80 mocked up because eun_bertology.py crashed
# RuntimeError: leaf variable has been moved into the graph interior
Data_MNLI_m = {
    'Percentage_Pruned': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Accuracy': [84, 83.6, 83.2, 82.4, 81.2, 78.4, 74.3, 68, 52.0, 45, 33.0]
}

df_anli_r1 = pd.DataFrame(Data_ANLI_r1)
print(df_anli_r1)

df_mnli_m = pd.DataFrame(Data_MNLI_m)
print(df_mnli_m)

plt.plot(df_anli_r1['Percentage_Pruned'],
         df_anli_r1['Accuracy'],
         '-o',
         label='ANLI-r1')

plt.plot(df_mnli_m['Percentage_Pruned'],
         df_mnli_m['Accuracy'],
         '-o',
         label='MNLI-m')

plt.axhline(y=33.3,
           linewidth=3,
           linestyle='--',
           color='r',
           label='Random')

plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Percentage pruned')
plt.grid(True)

plt.show()

# plt.savefig('percentage_pruned_vs_accuracy.eps')