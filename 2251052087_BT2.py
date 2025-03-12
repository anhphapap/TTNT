import pandas as pd
##PANDAS
#1
data = [4879,12104,12756,6792,142984,120536,51118,49528]
ds1 = pd.Series(data)
print(ds1)

#2
index=['Mercury', 'Venus', 'Earth', 'Mars', 'Jupyter', 'Saturn', 'Uranus', 'Neptune']
ds2 = pd.Series(data,index=index)
print(ds2)

#3
print(ds2['Earth'])

#4
print(ds2["Mercury":"Mars"])

#5
print(ds2[['Earth','Jupyter', 'Neptune']])

#6
ds2['Pluto'] = 2370

#7
data2 = {
    'diameter':[4879,12104,12756,6792,142984,120536,51118,49528,2370],
    'avg_temp':[167,464,15,-65,-110, -140, -195, -200, -225],
    'gravity':[3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7],
}
planets = pd.DataFrame(data2)
print(planets)

#8
print(planets[:3])

#9
print(planets.tail(2))

#10
print(planets.columns)

#11
index.append('Pluto')
planets.index = index
print(planets)

#12
print(planets["gravity"])

#13
print(planets[['gravity', 'diameter']])

#14
print(planets.loc['Earth','gravity'])

#15
print(planets.loc['Earth',['diameter','gravity']])

#16
print(planets.loc['Earth':'Saturn',['diameter','gravity']])

#17
print(planets[planets["diameter"] > 1000])

#18
print(planets[planets["diameter"] > 100000])

#19
print(planets[(planets['avg_temp']>0) & (planets['gravity'] > 5)])

#20
print(planets.sort_values("diameter",ascending=True))

#21
print(planets.sort_values("diameter",ascending=False))

#22
print(planets.sort_values("gravity",ascending=False))

#23
print(planets.sort_values(by='Mercury', axis=1))

##SEABORN
#1
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")
sns.set_style("whitegrid")
g = sns.lmplot(x="tip",y="total_bill", data=tips, aspect=2)
g = (g.set_axis_labels("Tip","Total bill(USD)").set(xlim=(0,10), ylim=(0,100)))
plt.title("title")
plt.show()

#2
# print(sns.get_dataset_names())

#3
# df = sns.load_dataset("tips")
# print(df)

#4
g = sns.lmplot(y="tip",x="total_bill", data=tips, aspect=2)
g = (g.set_axis_labels("Total bill(USD)","Tip").set(ylim=(0,10), xlim=(0,100)))
plt.show()

#5
sns.set_theme(font_scale=1.2, style="darkgrid")
sns.scatterplot(x='tip', y="total_bill", data=tips)
plt.show()

#6
sns.scatterplot(x='tip', y="total_bill", data=tips, hue='day')
plt.show()

#7
sns.scatterplot(x='tip', y="total_bill", data=tips,size='size', hue='day')
plt.show()

#8
g = sns.FacetGrid(tips, col='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip')
plt.show()

#9
g=sns.FacetGrid(tips, col='time', row='sex')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip')
plt.show()

## PANDAS
file_path="./04_gap-merged.tsv"

#1
df1 = pd.read_csv(file_path, nrows=5, sep='\t')
print(df1)

df = pd.read_csv(file_path, sep='\t')

#2
print(df.shape)

#3
print(df.columns)

#4
print(df.dtypes)

#5
country_col = df.country
print(country_col[:5])

#6
print(country_col.tail(5))

#7
df7=df[['country', 'continent', 'year']]
print(df7[:5], df7.tail(5))

#8
print(df.loc[[0]])
print(df.loc[[99]])

#9
print(df.iloc[:,[0]],df.iloc[:,[5]])

#10
print(df.iloc[[-1]])

#11
print(df.iloc[[0,99,999]])
print(df.loc[[0,99,999]])

#12
print(df.loc[42,['country']])
print(df.iloc[42,[0]])

#13
print(df.iloc[[0,99,999],[3,5]])

#14
df14 = df[:10]
print(df14)

#15
df15=df.loc[:,['lifeExp','year']].groupby('year').mean()
print(df15)

#16
print(df15[:5])

#17
s = pd.Series(data=['banana','42'], index=[0,1])
print(s)

#18
s = pd.Series(data=['banana','42','Wes MCKinney', 'Creator of Pandas'], index=[0,1,'Person','Who'])
print(s)

#19
data1 = {
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920-07-25', '1876-06-13'],
    'Died': ['1958-04-16', '1937-10-16'],
    'Age': [37, 61]
}

index1 = ['Franklin', 'Gosset']
ds = pd.DataFrame(data=data1, index=index1)
print(ds)