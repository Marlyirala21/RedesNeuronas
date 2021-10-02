import os

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# leemos y guardamos los datos
df = pd.read_csv('dataset.csv', header=0)

# normalizamos los datos con el método Min-max de cada columna del dataFrame
for col in df:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# aleatorizamos los datos
dfRandom = df.sample(frac=1)
train, validate, test = np.split(dfRandom, [int(.7*len(df)), int(.85*len(df))])

# creamos los archivos correspondientes
train.to_csv("train.csv")
validate.to_csv("validate.csv")
test.to_csv("test.csv")

def entrenamiento(wi,b):
        e = 0
        eFila = 0 #error actual de cada patrón/fila
        mse = 0 #error cuadrático medio
        eAnterior = 0
        eAcumulado = 0 #error acumulado de cada patrón

        for i in range(n):
            #calculamos la salida para cada patron/fila
            y = np.dot(entradas[i, :], wi) + b
            #ajuste de pesos y umbral/bias
            eFila = (d[i] - y) #error actual de cada fila
            wi = wi + (factAprendizaje * eFila * entradas[i,:]) #los nuevos pesos para la siguiente iteracion
            b = factAprendizaje * eFila
            # calculamos el error acumulado para posteriormente calcular el mse
            eAcumulado = eAcumulado + pow(eFila, 2)

            #calculamos el error cuadrático medio de cada ciclo
        mse = ((1/n) * eAcumulado)
        return wi, b, mse



def validacion(wAjustado, bAjustado):
        eFilaV=0
        mseV=0
        eAcumuladoV=0

        for i in range(nV):
            #calculamos la salida de cada patrón
            yV = np.dot(entradasV[i,:], wAjustado) + bAjustado
            #Ahora calculamos el error cuadratico medio
            eFilaV= dV[i]-yV
            eAcumuladoV = eAcumuladoV + pow(eFilaV, 2)
        mseV = ((1/nV) * eAcumuladoV)
        return mseV


def entrenamiento_Validacion(wi, b):
        ciclos=1500#numero de ciclos q quieres para la parada
        wi, bi, mseE = entrenamiento(wi, b)
        for i in range (ciclos):
            mseV = validacion(wi, b)
            eCMV.append(mseV)
            wi, bi, mseE = entrenamiento(wi, b)
            eCME.append(mseE)
        return eCME, eCMV

def fTest():
        eFilaT = 0
        mseT = 0
        eAcumuladoT = 0
        for i in range (nT):
            yT = np.dot(entradasT[i], wi) #************PREGUNTAR SI ESTA BIEN******************** #
            #Ahora calculamos el error cuadratico medio
            eFilaT= dT[i]-yT
            eAcumuladoT =eAcumuladoT + (pow(eFilaT,2))
            mseT = ((1/nV) * eAcumuladoT)

        print('el modelo tiene un eCM de test de:  ',str(mseT))
        return mseT

if __name__ == '__main__':
    #convertimos los datos en una matriz
    data = train.to_numpy()
    dataV = validate.to_numpy()
    dataT = test.to_numpy()
    #variables de entrada de los datos
    entradas = data[:, 0:8]
    entradasV = dataV[:, 0:8]
    entradasT = dataT[:, 0:8]
    #variable deseada
    d = data[:, 8]
    dV = dataV[:, 8]
    dT= dataT[:, 8]
    #vector pesos w
    wi = np.random.rand(8)
    #factor de aprendizaje
    factAprendizaje = 0.01
    #numero de muestras
    n = len(d) #721
    nV = len(dV)#154
    nT = len(dT)#155
    #bias
    b = random.uniform(0, 1)
    #ECM
    eCMV = []
    eCME = []

    errorE, errorV = entrenamiento_Validacion(wi,b)
    print("mseE: " + str(errorE) + "\nmseV: " + str(errorV))


    plt.plot(errorE, color='blue', marker='.', linewidth=2, markersize=10, label ='ECME')
    plt.plot(errorV, color='red', marker='.', linewidth=2, markersize=10, label ='ECME')
    plt.show()
     
###BIAS Y PESO EN VALOR ABSOLUTO
#QUE FACTOR DE APRENDIZAJE









