import pandas as pd
import seaborn as sns
import matplotlib. pyplot as plt
from sklearn. datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df.fillna(df.mean(), inplace=True)
df.drop_duplicates(inplace=True)
Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df< (Q1 - 1.5 * IQR)) | (df> (Q3 + 1.5 * IQR))).any(axis=1)]
print("=============== Preprocessed DataFrame ===============")
print(df.head())
sns.histplot(df['sepal length (cm)'], kde=True).set(title="Distribution of Sepal Length")
plt.show()
sns.scatterplot(x=df['sepal length (cm)'], y=df['sepal width (cm)']).set(title="Sepal Length vs Sepal Width")
plt.show()
print("✅ Data preprocessing completed successfully!")
