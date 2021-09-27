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

class Adaline():
    def __init__(self, wi, entradas , b, factAprendizaje, d, n, ciclos):
        self.wi = wi
        self.entradas = entradas
        self.b = b
        self.factAprendizaje = factAprendizaje
        self.d = d
        self.n = n
        self.ciclos = ciclos
        self.y = 0 #salida de la red

    #funcion de entrenamiento
    def entrenamiento(self):
        e = 0
        eFila = 0 #error actual de cada patrón/fila
        mse = 0 #error cuadrático medio
        eRed= []
        eAnterior = 0
        eAcumulado = 0 #error acumulado de cada patrón

        while (self.ciclos < 7):
            eAnterior = mse
            for i in range (self.n):
                #calculamos la salida para cada patron/fila
                self.y = np.dot(self.entradas[i], self.wi) + self.b #************PREGUNTAR SI ESTA BIEN********************
                #ajuste de pesos y umbral/bias
                eFila = (self.d[i] - self.y) #error actual de cada fila
                self.wi = self.wi + (self.factAprendizaje * eFila * self.entradas[i,:] )
                self.b = self.factAprendizaje * eFila
                # calculamos el error acumulado para posteriormente calcular el mse
                eAcumulado = eAcumulado + pow(eFila, 2)

            #calculamos el error cuadrático medio de cada ciclo
            mse = ((1/self.n) * (eAcumulado))
            e = (mse - eAnterior)
            eRed.append(np.abs(e))
            self.ciclos += 1
            print("este es el ciclo " + str(self.ciclos) + " " + " y tiene un error de: " + str(e))
        return self.wi, self.ciclos, eRed

"""DUDA: es aconsejable mostrar en pantalla el error medio a lo largo de los ciclos de aprendizaje.
Se refiere a ir mostrando el mse de cada ciclo o hay que calcular el error del mse de cada ciclo, 
es decir, hay que ir restando el mse actual al anterior y devolver la diferencia

DUDA 2: condicion de parada, como se haria lo de los ciclos
DUDA 3: todo lo del error de validacion

"""

if __name__ == '__main__':
    #convertimos los datos en una matriz
    data = train.to_numpy()
    #variables de entrada de los datos de entrenamiento
    entradas = data[:, 0:8]
    #variable deseada
    d = data[:, 8]
    #vector pesos w
    wi = np.random.rand(8)
    #w ajustado
    w = []
    #factor de aprendizaje
    factAprendizaje = 0.5
    #numero de muestras
    n = len(d) #721
    #bias
    b = random.uniform(0, 1)
    #ciclos
    ciclos = 0
    #inicializamos la red
    red = Adaline(wi, entradas , b, factAprendizaje, d, n, ciclos)
    wAjustado, ciclos, error = red.entrenamiento()
    #grafica
    fig, ax = plt.subplots()
    ciclos = np.arange(ciclos)
    error = {'error': error}
    ax.plot(ciclos, error['error'], marker = 'o')
    ax.set_title('Evolución del error', loc="left",
                 fontdict={'fontsize': 14, 'fontweight': 'bold', 'color': 'tab:blue'})
    plt.show()













    


