import os

import numpy as np
import pandas as pd
import random

# leemos y guardamos los datos
df = pd.read_csv('dataset.csv', header=0)

# normalizamos los datos con el método Min-max de cada columna del dataFrame
for col in df:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# aleatorizamos y separamos los datos
train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])

# creamos los archivos correspondientes
train.to_csv("train.csv")
validate.to_csv("validate.csv")
test.to_csv("test.csv")

class Adaline():
    def __init__(self, w, wAdjusted, x ,u,learnRate, concreteCompressiveStrength, n, epochs):
        self.w = w
        self.wAdjusted = wAdjusted
        self.x = x
        self.u = u
        self.learnRate = learnRate
        self.concreteCompressiveStrenght = concreteCompressiveStrength
        self.n = n
        self.epochs = epochs
        self.y = 0 #salida de la red

    #calculo de salida




if __name__ == '__main__':
    #convertimos los datos en una matriz
    data = train.to_numpy()
    #variables de entrada de los datos de entrenamiento
    x = data[:, 0:8]
    #variable deseada
    concreteCompressiveStrength = data[:, 8]
    #vector pesos w
    w = np.random.rand(8)
    #numero de muestras
    n = len(concreteCompressiveStrength)
    #factor de aprendizaje
    learnRate = 0.5
    #umbral
    u = random.uniform(0, 1)
    #ciclos
    epochs = 0
    #pesos ajustados
    wAdjusted = []
    #inicializamos la red
    net = Adaline( w, wAdjusted, x ,u,learnRate, concreteCompressiveStrength, n, epochs)





    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
