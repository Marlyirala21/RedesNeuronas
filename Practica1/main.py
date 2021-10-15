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
train.to_csv("train.csv", index = False)
validate.to_csv("validate.csv", index = False)
test.to_csv("test.csv", index = False)

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
    evMinimo = 1 #error validacion minimo
    cicloOptimo = 0
    pesosOptimos = []
    biasOptimo = 0
    ciclos = 1000
    numCiclo = []
    wi, bi, mseE = entrenamiento(wi, b)
    for i in range(ciclos):
        mseV = validacion(wi, b)
        eCMV.append(mseV)
        wi, bi, mseE = entrenamiento(wi, b)
        eCME.append(mseE)
        numCiclo.append(i)
        if mseV < evMinimo:
            evMinimo = mseV
            cicloOptimo = i
            pesosOptimos = wi
            biasOptimo = bi

    return eCME, eCMV, numCiclo, evMinimo, cicloOptimo, pesosOptimos, biasOptimo

def fTest(wOptimo,bOptimo): #wT y bT es el peso y bias que guardamos al hacer la validacion
        eFilaT = 0
        mseT = 0
        eAcumuladoT = 0
        salidasTest = []
        numPatron = []
        for i in range(nT):
            yTest = np.dot(entradasT[i], wOptimo) + bOptimo
            #Ahora calculamos el error cuadratico medio
            eFilaT= dT[i]-yTest
            eAcumuladoT = eAcumuladoT + (pow(eFilaT,2))
            salidasTest.append(yTest)
            numPatron.append(i)
        mseT = ((1/nT) * eAcumuladoT)
        return mseT, salidasTest, numPatron



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

    eCME, eCMV, numCiclo, evMinimo, cicloOptimo, pesosOptimos, biasOptimo = entrenamiento_Validacion(wi,b)
    eCMT, salidasTest, numPatron = fTest(pesosOptimos, biasOptimo)
    print(salidasTest)
    #creamos el fichero con las salidas de la red
    with open('salidasTest.txt', 'w') as yTest:
        for item in numPatron:
            yTest.write( "Patron " + str(item) + ": " + "%s\n" % salidasTest[item])


    #creamos un dataframe para ver los diferentes errores de entrenamiento y validacion en cada ciclo
    df = pd.DataFrame(list(zip(numCiclo,eCME, eCMV)), columns = ['numCiclos', 'errorEntrenamiento','errorValidacion'])

    #imprimimos los valores
    print(df)
    print("El nº de ciclos óptimos es: " + str(cicloOptimo))
    print("Los pesos óptimos son: " + str(pesosOptimos))
    print("El bias óptimo es: " + str(biasOptimo))
    print("El error de test es: " + str(eCMT))

    # gráfica
    plt.plot(eCME, color='blue', marker='.', linewidth=2, markersize=0, label='ECME')
    plt.plot(eCMV, color='red', marker='.', linewidth=2, markersize=0, label='ECMV')
    plt.show()













