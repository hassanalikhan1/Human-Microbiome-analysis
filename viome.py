#!/usr/bin/env python
# coding: utf-8

# # Human Microbiome Analysis
# 
# This project is based on the dataset containing the simulated microbiome data of customers and the target variable, which indicates whether the person has disease X or not. The dataset consisted of the userIDs, RNA read counts mapped to 1000 microbe's genome. The code goes in the following sequence.
# 
# 1) Importing the required libraries
# 2) Data analysis including cleaning and visualization
# 3) Data modelling

# In[4]:


import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")


import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



# In[5]:


# reading the data using pandas library
df = pd.read_csv("data_labels_user_id.csv")
df.head()


# DATA EXPLORATION AND CLEANING:
# 
# 1) Checked if the data consists of any null values
# 2) Extracted the features (RNA read count)
# 3) Calculated the relative proportion of each microbe by divding the RNA read count by the total of reads in the sample
# 4) Normalized the data to maintain the general distribution and ratios in the source data, while keeping values within a scale applied across all numeric columns, in this case between 0 and 1, used in the model.
# 5) Dimensionality reduction to extract the important features
# 6) Data visualization

# In[6]:


df.describe()


# In[7]:


df.info()


# In[24]:


#Now,I will check null on all data and If data has null
print('Data Sum of Null Values \n')
df.isnull().sum()


# The data contains 0 null values

# In[9]:


## get the features without the userid and label
features=df.values[:,1:1001]
labels=df.values[:,-1]
user_ids=df.values[:,0]


# In[10]:


##calculate the relative RNA read count values of all the microbes to 
sumFeatures=np.sum(features,axis=1)
relat_features= [features[i,:]/sumFeatures[i] for i in range(0,features.shape[0])]


# In[11]:


# In order to ensure the data is normalized, I used the sklearn MinMaxScaler function to convert each feature value between 
# 0 and 1
scaler=preprocessing.MinMaxScaler(feature_range=(0,1))
vals=scaler.fit_transform(features)
df_features=pd.DataFrame(vals)
df_features.head()


# In[12]:


##group the values with respect to the labels,0 and 1 to explore the data
labels=pd.DataFrame(labels,columns=['labels'],dtype='int')
df_features1=pd.concat([df_features,labels],axis=1)
df_features1.groupby('labels').mean()


# Dimensionality reduction:
# 
# Since this is a high-dimensional data, we will have to perform dimensionality reduction techniques for a better model perfomance. Curse of dimensionality refers to an exponential increase in the size of data caused by a large number of dimensions. As the number of dimensions of a data increases, it becomes more and more difficult to process it. Dimension Reduction is a solution to the curse of dimensionality.
# 
# In order to solve this problem, we will use PCA and sklearn filter selection method.PCA is a projection based method which transforms the data by projecting it onto a set of orthogonal axes.
# The code is give below.

# In[13]:


##performing feature selection to determine which features are important for our model using chi sqaure method
FSelectModel=SelectKBest(score_func=chi2,k=10)

selectF=pd.DataFrame(FSelectModel.fit_transform(df_features1.values[:,:1000],df_features1.labels.values.tolist()))
newFeatures=pd.DataFrame(selectF, dtype=float)
print (newFeatures.head())


# We will be using the PCA ANALYSIS to project the dataset in lower dimensions to find obvious cluster boundaries. I will be using sklearn PCA function to apply to the transformed data and plot the data in two dimensions. This is only for the purpose of visualization.
# 

# In[14]:


pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(df_features.values))
transformed1=pd.concat([transformed,labels],axis=1)
# transformed.columns=['feature1','feature2']

print(transformed1[transformed1.labels==0].shape)

fig = plt.figure()
ax=plt.axes()

ax.scatter(transformed1[transformed1['labels']==0][0], transformed1[transformed1['labels']==0][1], label='Class 0', c='blue')
ax.scatter(transformed1[transformed1['labels']==1][0], transformed1[transformed1['labels']==1][1], label='Class 1', c='red')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatterplot of PCA features of both classes')
plt.legend()
plt.show()


# The scatter plot above shows the plot of values in the new feature space of dimension 2.PCA creates a new set of features that is a linear combination of the input features As we can see that apart from the few outliers, the feature values of class 0 and 1 almost perfectly coincide. This shows that each principal component was a linear combination of highly uncorelated values.

# In[15]:


##drawing the countplot of the labels column
sns.countplot(x='labels',data=df_features1,palette='bwr')


# The number of users with disease X are higher in number than the number of users with no disease X.

# In[16]:


##plot the means of each target
new_features=pd.DataFrame(pd.concat([newFeatures,labels],axis=1))

meanLabels=new_features.groupby('labels').mean()
plt.plot(meanLabels.values[0,1:],label='Class 0')
plt.plot(meanLabels.values[1,1:],label='Class 1')

plt.title('Mean of features of both classes')
plt.xlabel("Microbe number")
plt.ylabel("RNA read count ")

plt.legend()
plt.show()


# The above line graph shows the mean RNA counts (new selected features) of both classes.We can see that the average RNA read count of microbes of patients with disease is compartively lower than the RNA count of microscobes of patients without disease.

# In[17]:


##Make a bar graph comparing RNA counts of patients with and without disease
meanLabels.plot.bar(title='Bar graph of microbes RNA read count of both classes')
plt.xlabel("Class ")
plt.ylabel("RNA read count ")


# This bar graphs compares the mean feature (RNA read counts) values of class 0 and 1. We can see that for each microbic genome, RNA read count of patients with disease is lower than those without disease.

# In[18]:


corr = newFeatures.corr()
corr.style.background_gradient(cmap='coolwarm')


# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. A correlation matrix is used as a way to summarize data, as an input into a more advanced analysis, and as a diagnostic for advanced analyses.
# 
# Our corelation matrix shows that our data is strongly not correlated. This indicates that our features are non-redundant,which is important for model's performance.
# 

# In[19]:


##
meanLabels.plot.box()
plt.title("Boxplot of all microbes")
plt.xlabel("Microbe number ")
plt.ylabel("RNA read count ")


# Data Modelling:
#     1) Separated the training andd the testing data with 80/20 ratio.
#     2) Used Logistic Regression, Decision tree, adaboost and multi-layer perceptron classifer in our model.

# In[20]:


x_train, x_test, y_train, y_test = train_test_split(df_features.values,labels,test_size = 0.2,random_state=0)
print(x_train.shape)

##USing Logistic Regression
logR = LogisticRegression(random_state=0, solver='lbfgs').fit(x_train, y_train)
predicted=logR.predict(x_test)
print("Logistic Regression acuuracy: " + str(logR.score(x_test,y_test)))

#USING DECiSION TREE CLASSIFiER

dt=DecisionTreeClassifier(max_depth=5)
dt.fit(x_train,y_train)
accur=dt.score(x_test,y_test)
print("Decision Tree Classifier: " + str(accur))

# ## USING ADABOOST CLASSIFER

ad=AdaBoostClassifier()
ad.fit(x_train,y_train)
accur=ad.score(x_test,y_test)
print("Adaboost Classifier: " + str(accur))

#USING MULTI-lAYER PERCEPTRON
mlp=MLPClassifier(alpha=1)
mlp.fit(x_train,y_train)
accur=mlp.score(x_test,y_test)
print("MULTI-lAYER PERCEPTRON Classifier: " + str(accur))



# Our results show that that Logistic Regression CLassifier performs best with 80.6% accuracy. We can predict that Logistic regression is a perfectly regression algorthim for a spurse uncoorelated data like this because it is more robust: the independent variables don’t have to be normally distributed, or have equal variance in each group and it can handle non-linearality pretty well.
# 

# In[21]:


logScore=cross_val_score(logR, x_test, y_test, scoring='recall_macro',cv=5)
plt.plot(logScore)
plt.xlabel('training instances')
plt.ylabel('Cross-validation score')
plt.title('Logistic Regression performance')


# Our results show that the model's performance is decreasing because of overfitting of data. Overfitting is a modeling error which occurs when a function is too closely fit to a limited set of data points. 

# In[22]:


##this code has been taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
  
    print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[23]:


plot_confusion_matrix(y_test, predicted, classes=labels,
                      title='Confusion matrix, without normalization')


# A confusion matrix is a summary of prediction results on a classification problem.
# The number of correct and incorrect predictions are summarized with count values and broken down by each class. In this case, true positives are greater than false positives and true negatives are greater than false negatives.
