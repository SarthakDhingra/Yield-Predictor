#import data anlysis libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from fancyimpute import KNN, IterativeImputer

#import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#read dataset
dataset = pd.read_csv("dataset.csv")

#encode categorical data by pesticide type
dataset['categories'] = dataset['categories'].str.split(',')
dataset['pa'] = 0
dataset['pb'] = 0
dataset['pc'] = 0
dataset['pd'] = 0

length = dataset.index

for values in length:
    if 'a' in dataset.iloc[values][8]:
        dataset.iloc[values, 9] = 1
    if 'b' in dataset.iloc[values][8]:
        dataset.iloc[values, 10] = 1
    if 'c' in dataset.iloc[values][8]:
        dataset.iloc[values, 11] = 1
    if 'd' in dataset.iloc[values][8]:
        dataset.iloc[values, 12] = 1

#remove categories column
dataset.pop('categories')

#encode categorical data by region
dataset['r0'] = 0
dataset['r1'] = 0
dataset['r2'] = 0
dataset['r3'] = 0
dataset['r4'] = 0
dataset['r5'] = 0
dataset['r6'] = 0

for values in length:
    if dataset.iloc[values][7] == 0:
        dataset.iloc[values, 12] = 1
    if dataset.iloc[values][7] == 1:
        dataset.iloc[values, 13] = 1
    if dataset.iloc[values][7] == 2:
        dataset.iloc[values, 14] = 1
    if dataset.iloc[values][7] == 3:
        dataset.iloc[values, 15] = 1
    if dataset.iloc[values][7] == 4:
        dataset.iloc[values, 16] = 1
    if dataset.iloc[values][7] == 5:
        dataset.iloc[values, 17] = 1
    if dataset.iloc[values][7] == 6:
        dataset.iloc[values, 18] = 1

#remove region column
dataset.pop('region')

#create scatterplots to visually identify outliers and check linearity
features = ['water','uv','area','fertilizer_usage','pesticides']
for feature in features:
    sns.scatterplot(x=feature,y='yield', data=dataset)
    plt.show()
    plt.clf()

#remove outlier seen in visual inspection
max = dataset['water'].max()
dataset = dataset[dataset.water != max]
max = dataset['water'].max()

#impute missing data
#data = IterativeImputer().fit_transform(dataset)
data = KNN(k=21).fit_transform(dataset)

#remove id as its not relevant to regression
fill_df = pd.DataFrame({
                        'water':data[:,1],'uv':data[:,2],'area':data[:,3],'fertilizer_usage':data[:,4],'yield':data[:,5],
                        'pesticides':data[:,6],'pa':data[:,7],'pb':data[:,8],'pc':data[:,9],'pd':data[:,10],
                        'r0':data[:,11],'r1':data[:,12],'r2':data[:,13],'r3':data[:,14],'r4':data[:,15],'r5':data[:,16],'r6':data[:,17]
                        })

#create squared and cubed columns of continuous data
fill_df['water2'] = fill_df['water']**2
fill_df['uv2'] = fill_df['uv']**2
fill_df['pesticides2'] = fill_df['pesticides']**2

fill_df['water3'] = fill_df['water']**3
fill_df['uv3'] = fill_df['uv']**3
fill_df['pesticides3'] = fill_df['pesticides']**3

#create scatterplot of data
features = ['water','uv','area','fertilizer_usage','pesticides','water2','uv2','pesticides2','water3','uv3','pesticides3']
for feature in features:
    scatter = sns.scatterplot(x=feature,y='yield', data=fill_df)
    scatter.figure.savefig(feature + '.png')
    plt.clf()

#create final
fill_df.to_csv('Final.csv')

#calculate correlation coefficients of data and output to file
output = open('output.txt','w+')
features = ['water','uv','area','fertilizer_usage','pesticides','water2','uv2','pesticides2','water3','uv3','pesticides3','pa','pb','pc','pd','r0','r1','r2','r3','r4','r5','r6']
for values in features:
    correlations = fill_df[values].corr(fill_df['yield'])
    output.write("Yield vs. " + values + ": " + str(correlations) + "\n")

#create training and test split
y = fill_df.pop('yield')
x = fill_df
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

#create regression model
mlr = LinearRegression(normalize = True)
regressor = mlr.fit(x_train,y_train)

#test accuracy of regression model
accuracies = []
for values in range(10000):
    accuracy = regressor.score(x_test,y_test)
    accuracies.append(accuracy)
correct = 100*(sum(accuracies)/10000)
output.write('The accuracy of the model is ' + str(correct) + '%')
exit()
