import seaborn as sns
import numpy as np
tips=sns.load_dataset('tips')
tips.head()
# non numeric columns are called categorical columns like sex, smoker, day,time
#bar plot with one categorical values exand another non categorical value total_bill
sns.barplot(x='sex',y='total_bill',data=tips)
# default estimator is mode, we can change it as median
sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.median)
#Count Plot
# pass single data to count plot
sns.countplot(x='sex',data=tips)
tips['sex'].value_counts()
sns.countplot(y='sex',data=tips)
#we can make meaning ful split of the data by adding hue with other fields
sns.countplot(x='sex',data=tips,hue='day')

sns.countplot(x='sex',data=tips,hue='time')
#boxplot
sns.boxplot(x='sex',y='total_bill',data=tips)
sns.violinplot(x='sex',y='total_bill',data=tips)
sns.violinplot(x='smoker',y='total_bill',data=tips)
sns.violinplot(x='sex',y='total_bill',hue='smoker',data=tips)
# New Section
sns.stripplot(x='sex',y='total_bill',data=tips)
sns.stripplot(x='sex',y='total_bill',hue='day',data=tips)
sns.stripplot(x='sex',y='total_bill',hue='smoker',data=tips)
sns.swarmplot(x='sex',y='total_bill',data=tips)
sns.swarmplot(data=tips,x='day',y='tip')
sns.swarmplot(x='day',y='tip',hue='sex',data=tips)
sns.lineplot(x="sex",y="total_bill",data=tips)
sns.pointplot(x="sex",y="total_bill",data=tips)
sns.relplot(x="sex",y="total_bill",data=tips)
sns.scatterplot(x="sex",y="total_bill",data=tips)
sns.kdeplot(y="total_bill",data=tips)
sns.pairplot(data=tips)
sns.displot(tips, x="size")
