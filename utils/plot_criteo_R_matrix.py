import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# change the directory to your settings
flights = pd.read_csv('//Users//wayne//Desktop//criteo_R.csv', header=None)

flights.index += 1 
flights.columns += 1




plt.figure(figsize=(8, 7))
#sns.heatmap(flights, vmin=-1.1, vmax=1.1, linewidths=.0005, cmap=sns.color_palette("BrBG", 100), xticklabels=7, yticklabels=7)
sns.heatmap(flights, vmin=-1.1, vmax=1.1, linewidths=.0005, cmap=sns.color_palette("BrBG", 100), xticklabels=False, yticklabels=False)


#sns_plot.savefig("criteo_r.png")

plt.savefig('//Users//wayne//Desktop//criteo_r.png', figsize=(8, 7))
