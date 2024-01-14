import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

data=pd.read_excel(r"C:/Users/PC/OneDrive/Desktop/project.xlsx")
pd.set_option('display.max_rows',None)
print(data)
data['Crop Recommendation'].value_counts()
encoder=LabelEncoder()
for column in data.columns:
    if data[column].dtypes=='object':
        data[column]=encoder.fit_transform(data[column])
print(data)
data['Crop Recommendation'].value_counts()
data.head()
X=data.drop('Crop Recommendation',axis=1)
y=data['Crop Recommendation']
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model= RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predicted_targets = model.predict(X_test)
accuracy=accuracy_score(y_test,predicted_targets)
precision = precision_score(y_test, predicted_targets, average='micro')
recall = recall_score(y_test, predicted_targets, average='micro')
f1 = f1_score(y_test, predicted_targets, average='micro')
print('Accuracy:', accuracy)
print('''
0 for alluvial  
1 for alluvium
2 for black 
3 for black soil
4 for clay
5 for clay loam  
6 for gravels   
7 for  loam  
8 for loamy sand 
9 for loamy soil    
10 for medium black 
11 for mixed red and black   
12 for red 
13 for red loamy   
14 for red sandy 
15 for sandy clay
16 for sandy loam 
17 for sandy-clay loam              
18 for shallow black              
        ''')
i=(input("Select the soil type and enter the corrosponding number of the soil by refering the above soil chart\n:"))
j=(input("enter the ph of the soil\n:"))
k=(input("enter the temperature\n:"))
new_data = [[i,j,k]]
predicted_crop = model.predict(new_data)
if i==0:
  print("Soil Type:alluvial")
elif i==1:
  print(" Soil Type:alluvium")  
elif i==2:
  print(" Soil Type:black")  
elif i==3:
  print(" Soil Type:black soil")  
elif i==4:
  print(" Soil Type:clay")  
elif i==5:
  print(" Soil Type:clay loam")  
elif i==6:
  print(" Soil Type:gravels")  
elif i==7:
  print(" Soil Type:loam")  
elif i==8:
  print(" Soil Type:loamy sand")  
elif i==9:
  print(" Soil Type:loamy soil")          
elif i==10:
  print(" Soil Type:medium black")  
elif i==11:
  print(" Soil Type:mixed red and black")  
elif i==12:
  print(" Soil Type:red")  
elif i==13:
  print(" Soil Type:red loamy")  
elif i==14:
  print(" Soil Type:red sandy")  
elif i==15:
  print(" Soil Type:sandy clay")              
elif i==16:
  print(" Soil Type:sandy loam")              
elif i==17:
  print(" Soil Type:sandy-clay loam")              
elif i==18:
  print(" Soil Type:shallow black")              
print("Soil Ph",j)
print("Temperature",k)
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1 score:', f1)
# print(predicted_crop)
if predicted_crop==13:
  print(" the recommended crop is rice")
elif predicted_crop==1:
  print("the recommended crop is Apple")
elif predicted_crop==2:
  print("the recommended crop is Banana")
elif predicted_crop==3:
  print("the recommended crop is Beans")
elif predicted_crop==4:
  print("the recommended crop is coffee")
elif predicted_crop==5:
  print("the recommended crop is Cotton")
elif predicted_crop==6:
  print("the recommended crop is Cowpeas")
elif predicted_crop==7:
  print("the recommended crop is grapes")
elif predicted_crop==8:
  print("the recommended crop is groundnut")
elif predicted_crop==9:
  print("the recommended crop is maiz")
elif predicted_crop==10:
  print("the recommended crop is mango")
elif predicted_crop==11:
  print("the recommended crop is Orange")
elif predicted_crop==12:
  print("the recommended crop is peas")
elif predicted_crop==14:
  print("the recommended crop is Watermelon")
elif predicted_crop==0:
  print("the recommended crop is Soyabean")