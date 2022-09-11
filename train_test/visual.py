import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

dataFrame = pd.read_csv("E:/4-9-2022/dev1/cropped/reject/result.csv")

# plotting scatterplot with Age and Weight (kgs)
# hue parameter set as "Role"
sb.scatterplot(dataFrame['ASM'])

plt.ylabel("Weight (kgs)")
plt.show()