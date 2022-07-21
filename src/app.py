# Aca ponemos el codigo para llegar al mismo resultado de explore.ipynb
# Esto es un trabajo que resumimos las consultas que vamos haciendo en el cuaderno

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

df=pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv')

print(df.shape)
corr_matrix = df.drop(columns=['ICU Beds_x']).corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop features 
df.drop(to_drop, axis=1, inplace=True)

df=df.drop(columns=['COUNTY_NAME'])

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['STATE_NAME'] = encoder.fit_transform(df['STATE_NAME'])

X= df.drop(columns=['ICU Beds_x'])
y =df['ICU Beds_x']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=94)

pipe_lr = make_pipeline(StandardScaler(), Lasso())
pipe_lr.fit(X_train, y_train)  
pipe_lr.score(X_test, y_test)

y_train_pred = pipe_lr.predict(X_train)

rms_train = mean_squared_error(y_train, y_train_pred, squared=False)

print("RMS train =",rms_train)

y_test_pred = pipe_lr.predict(X_test)

rms_test= mean_squared_error(y_test, y_test_pred, squared=False)

print("RMS test =",rms_test)

# Tengo que cambiar los valores de los parametros para mejorar los valores
# Pruebo con alpha = 0.8 pero no cambia mucho el resultado

pipe_lr = make_pipeline(StandardScaler(), Lasso(alpha=2))
pipe_lr.fit(X_train, y_train)  
pipe_lr.score(X_test, y_test)

def funcion_prueba(valor):
    pipe_lr = make_pipeline(StandardScaler(), Lasso(alpha=valor))
    pipe_lr.fit(X_train, y_train)  
    y_test_pred = pipe_lr.predict(X_test)

    rms_test= mean_squared_error(y_test, y_test_pred, squared=False)
    return rms_test

# Con valores de rango para alfa entre 1 y 9
menor=100
indice=0
for i in range(1,10):
    dato=funcion_prueba(i)
    
    if dato<menor:
        menor=dato
        indice=i
    print(f'El valor calculado para alfa ={i} es {dato}')  

print("-"*70)
print(f'El valor mas bajo es para alfa ={indice} y {menor}')   


# Con valores para alfa entre 0.0 y 0.9
menor=100
indice=0
seq = (x/10 for x in range(0, 10))
for i in seq:
    dato=funcion_prueba(i)
    
    if dato<menor:
        menor=dato
        indice=i
    print(f'El valor calculado para alfa ={i} es {dato}')  

print("-"*70)
print(f'El valor mas bajo es para alfa ={indice} y {menor}')

# Visto las dos ejecuciones vemos que para alfa =1 es el menor valor de RMSE

# este seria el mejor caso

pipe_lr = make_pipeline(StandardScaler(), Lasso(alpha=1))
pipe_lr.fit(X_train, y_train)  
y_test_pred = pipe_lr.predict(X_test)

rms_test= mean_squared_error(y_test, y_test_pred, squared=False)

y_train_pred = pipe_lr.predict(X_train)
rms_train = mean_squared_error(y_train,y_train_pred, squared=False)

print("rms train",rms_train)
print("rms test",rms_test)

pickle.dump(pipe_lr, open('../models/lasso.pkl', 'wb'))

print("El archivo entrenado se ha guardado correctamente")
print("Dentro de la carpeta de models")
print()
print("Fin del proceso y del programa")


