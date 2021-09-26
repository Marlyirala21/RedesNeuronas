import os

import numpy as np
import pandas as pd
import random
import matplotlip as plt

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

class Adaline():
    def __init__(self, wi, entradas , b, factAprendizaje, d, n, ciclos, errorAceptado):
        self.wi = wi
        self.entradas = entradas
        self.b = b
        self.factAprendizaje = factAprendizaje
        self.d = d
        self.n = n
        self.ciclos = ciclos
        self.errorAceptado = errorAceptado
        self.y = 0 #salida de la red

    #funcion de entrenamiento
    def entrenamiento(self):
        e = 1
        eAnterior = 0
        eFila = 0 #error actual de cada patrón/fila
        mse = 0 #error cuadrático medio
        eRed= []
        eAcumulado = 0 #error acumulado de cada patrón

        while (np.abs(e) > self.errorAceptado):
            eAnterior = mse
            for i in range (self.n):
                #calculamos la salida para cada patron/fila
                self.y = np.dot(entradas[i], wi) + b
                #ajuste de pesos y umbral/bias
                self.eFila = (self.d[i] - self.y) #error actual de cada fila
                self.wi = self.wi + (self.factAprendizaje * eFila * entradas[i,:] )
                self.b = self.factAprendizaje * eFila
                # calculamos el error acumulado para posteriormente calcular el mse
                self.eAcumulado = eAcumulado + pow(eFila,2)

            #calculamos el error cuadrático medio
            mse = ((1/self.n) * (self.eAcumulado))
            e = mse - eAnterior
            eRed.append(np.abs(e))
            self.ciclos += 1
        return self.wi, self.ciclos, eRed

"""DUDA: es aconsejable mostrar en pantalla el error medio a lo largo de los ciclos de aprendizaje.
Se refiere a ir mostrando el mse de cada ciclo como esta ahora o hay que calcular el error de cada ciclo, 
es decir, habria que ir añadiendo -> e = mse - mseAnterior (siendo msePrevio = mse al principio del ciclo)

"""




if __name__ == '__main__':
    #convertimos los datos en una matriz
    data = train.to_numpy()
    #variables de entrada de los datos de entrenamiento
    entradas = data[:, 0:8]
    d = data[:, 8]#variable deseada
    #vector pesos w
    wi = np.random.rand(8)
    #w ajustado
    w = []
    # factor de aprendizaje
    factAprendizaje = 0.5
    #numero de muestras
    n = len(d) #721
    #bias
    b = random.uniform(0, 1)
    #ciclos
    ciclos = 0
    #error aceptado
    errorAceptado = 0.001
    #inicializamos la red
    red = Adaline(wi, entradas , b, factAprendizaje, d, n, ciclos, errorAceptado)
    wAjustado, ciclos, error = red.entrenamiento()

    print(ciclos)










    

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
