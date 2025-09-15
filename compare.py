import matplotlib.pyplot as plt
import pandas as pd

from lr import run as run_lr
from dt import run as run_dt
from rf import run as run_rf
from Svm import run as run_svm

models = [
    run_lr(show_plot=False, verbose=False),
    run_dt(show_plot=False, verbose=False),
    run_rf(show_plot=False, verbose=False),
    run_svm(show_plot=False, verbose=False),
]

df = pd.DataFrame(models)
df = df[["model", "accuracy", "f1", "auc"]]
df = df.sort_values(by="auc", ascending=False).reset_index(drop=True)

print("=== Model Comparision (By AUC) ===")
print(df.to_string(index=False, float_format='{:0.4f}').format)

df.to_csv("compare.csv", index=False)

df_plot = df.set_index("model")
df_plot = [["auc", "f1", "accuracy"]].plot_bar(rot=45)
plt.title("Model Comparison")
plt.tight_layout()
plt.show()