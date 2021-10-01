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
    def __init__(self, wi, entradas, entradasV, entradasT, b, factAprendizaje, d, dV, dT, n, nV, nT, ciclos, eCMV):      #wi= pesos, #n= numero de filas(iteraciones), #b = bias, #d= salida deseada
        self.wi = wi
        self.entradas = entradas
        self.entradasV = entradasV
        self.b = b
        self.factAprendizaje = factAprendizaje
        self.d = d
        self.dV = dV
        self.dT = dT
        self.n = n
        self.nV = nV
        self.nT = nT
        self.ciclos = ciclos
        self.y = 0 #salida de la red
        self.yV = 0
        self.eCMV = eCMV
        self.entradasT = entradasT

    #funcion de entrenamiento
    def entrenamiento(self): # se ebtrena un solo ciclo ya q luego hay q hacer validacion
        e = 0
        eFila = 0 #error actual de cada patrón/fila
        mse = 0 #error cuadrático medio
        eRed= []
        eAnterior = 0
        eAcumulado = 0 #error acumulado de cada patrón

        #while (self.ciclos < 7):
        #    eAnterior = mse

        for i in range (self.n):
            #calculamos la salida para cada patron/fila
            self.y = np.dot(self.entradas[i], self.wi) + self.b #************PREGUNTAR SI ESTA BIEN********************
            #ajuste de pesos y umbral/bias
            eFila = (self.d[i] - self.y) #error actual de cada fila
            #print('los pesos para la iteracion  ', i, '  son  ', self.wi)###################################################
            self.wi = self.wi + (self.factAprendizaje * eFila * self.entradas[i,:] )#los nuevos pesos para la siguiente iteracion
            self.b = self.factAprendizaje * eFila
            #print('los pesos de cada iteracion de entrenamiento son:   ', self.wi)####################
            # calculamos el error acumulado para posteriormente calcular el mse
            eAcumulado = eAcumulado + pow(eFila, 2)

            #print('los primeros pesos del ciclo  ',self.ciclos+1,  '  son  ', self.wi)#################################################################
            #calculamos el error cuadrático medio de cada ciclo
        mse = ((1/self.n) * eAcumulado)
        e = (mse - eAnterior) # el error cuadratico medio de cada ciclo
        eRed.append(np.abs(e)) #lista con el error cuadratico medio de todos los ciclos
        self.ciclos += 1
        print("este es el ciclo " + str(self.ciclos) + " " + " y tiene un error cuadratico medio de: " + str(e), ' para los datos de entrenamiento')
        return self.wi, self.ciclos, eRed

    #self.wi son los ultimos pesos q se han ido guardando en self.wi


    def validacion(self):
        eFilaV=0
        mseV=0
        eAcumuladoV=0
        for i in range (self.nV):
            self.yV = np.dot(self.entradasV[i], self.wi) #************PREGUNTAR SI ESTA BIEN******************** # le he quitado el bias pq si no, siempre crecia su valor
            #en la linea de arriba estamos obteniendo las salidas de las filas de Validacion siempre con los mismos pesos, obtenidos del final de cada ciclo de entrenamiento
            #Ahora calculamos el error cuadratico medio
            eFilaV= self.dV[i]-self.yV
            eAcumuladoV =eAcumuladoV + (pow(eFilaV,2))
            mseV = ((1/self.nV) * eAcumuladoV)
        print("este es el ciclo " + str(self.ciclos) + " " + " y tiene un error cuadratico medio de: " + str(mseV), ' para los datos de validacion')
        #print('los pesos q se usan para la primera validacion son', self.wi )######################

        return mseV


    def entrenamiento_Validacion(self):
        x=3 #numero de ciclos q quieres para la parada
        for i in range (x):
            red.entrenamiento()
            a = red.validacion()
            eCMV.append(a)
        r = True
        while (r):
            red.entrenamiento()
            a = red.validacion()
            eCMV.append(a)
            #ultimo en lista        #penultimos           #antepenultimo
            if ((eCMV[-(x-2)] == eCMV[-(x-1)]) & (eCMV[-(x-1)] == eCMV[-x])):
                r = False
                print('se acabo el loop pq los 3 ultimos eCMVs son iguales',eCMV[-(x-2)],eCMV[-(x-1)],eCMV[-(x)])
            if((eCMV[-(x-2)] > eCMV[-(x-1)]) & (eCMV[-(x-1)] > eCMV[-x])):
                r = False
                print('se acabo el loop pq los 3 ultimos eCMVs van en aumento',eCMV[-(x)],eCMV[-(x-1)],eCMV[-(x-2)])



    def test(self):
        eFilaT = 0
        mseT = 0
        eAcumuladoT = 0
        for i in range (self.nT):
            self.yT = np.dot(self.entradasT[i], self.wi) #************PREGUNTAR SI ESTA BIEN******************** #
            #Ahora calculamos el error cuadratico medio
            eFilaT= self.dT[i]-self.yT
            eAcumuladoT =eAcumuladoT + (pow(eFilaT,2))
            mseT = ((1/self.nV) * eAcumuladoT)

        print('el modelo tiene un eCM de test de:  ',str(mseT))
        return mseT



"""DUDA: es aconsejable mostrar en pantalla el error medio a lo largo de los ciclos de aprendizaje.
Se refiere a ir mostrando el mse de cada ciclo o hay que calcular el error del mse de cada ciclo, 
es decir, hay que ir restando el mse actual al anterior y devolver la diferencia
"""

if __name__ == '__main__':
    #convertimos los datos en una matriz
    data = train.to_numpy()
    dataV = validate.to_numpy()
    dataT = test.to_numpy()
    #variables de entrada de los datos de entrenamiento
    entradas = data[:, 0:8]
    entradasV = dataV[:,0:8]
    entradasT = dataT[:,0:8]
    #variable deseada
    d = data[:, 8]
    dV = dataV[:, 8]
    dT= dataT[:,8]
    #vector pesos w
    wi = np.random.rand(8)
    #w ajustado
    w = []
    #factor de aprendizaje
    factAprendizaje = 0.5
    #numero de muestras
    n = len(d) #721
    nV = len(dV)#154
    nT = len(dT)#155
    #bias
    b = random.uniform(0, 1)
    #ciclos
    ciclos = 0
    #ECM
    eCMV = []
    #inicializamos la red
    red = Adaline(wi, entradas, entradasV,entradasT, b, factAprendizaje, d, dV, dT, n, nV, nT, ciclos, eCMV)
    #wAjustado, ciclos, error = red.entrenamiento()############################################################################################################################
    red.entrenamiento_Validacion()
    red.test()

     # Aqui estamos guardando el ECM de la validacion en cada ciclo para luego usarlo como condicion de parada
    #wAjustado = los ultimos pesos de la red(ultimo ciclo ultima iteracion y actualizados)
    #print('los ciclos son  ',ciclos,'El wAjustado es  ',wAjustado, 'El error es  ', error)#################################
    #grafica
    '''fig, ax = plt.subplots()
    ciclos = np.arange(ciclos)
    error = {'error': error}
    ax.plot(ciclos, error['error'], marker = 'o')
    ax.set_title('Evolución del error', loc="left",
                 fontdict={'fontsize': 14, 'fontweight': 'bold', 'color': 'tab:blue'})
    plt.show()
    '''






