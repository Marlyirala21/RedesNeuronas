import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt




def entrenamiento(wi,b):
        eFila = 0 #error actual de cada patrón/fila
        mse = 0 #error cuadrático medio
        eAcumulado = 0 #error acumulado de cada patrón

        for i in range(nE):
            #calculamos la salida para cada patron/fila
            y = np.dot(entradasE[i, :], wi) + b
            #ajuste de pesos y umbral/bias
            eFila = (valorDeseadoE[i] - y) #error actual de cada fila
            wi += (factAprendizaje * eFila * entradasE[i,:]) #los nuevos pesos para la siguiente iteracion
            b = factAprendizaje * eFila
            # calculamos el error acumulado para posteriormente calcular el mse
            eAcumulado += pow(eFila, 2)

        #calculamos el error cuadrático medio de cada ciclo
        mse = ((1/nE) * eAcumulado)
        return wi, b, mse



def validacion(wAjustado, bAjustado):
        eFilaV = 0
        mseV = 0
        eAcumuladoV = 0
        for i in range(nV):
            #calculamos la salida de cada patrón
            yV = np.dot(entradasV[i,:], wAjustado) + bAjustado
            #Ahora calculamos el error cuadratico medio
            eFilaV = valorDeseadoV[i]-yV
            eAcumuladoV += pow(eFilaV, 2)
        mseV = ((1/nV) * eAcumuladoV)
        return mseV


def fTest(wOptimo,bOptimo): #wT y bT es el peso y bias que guardamos al hacer la validacion
        eFilaT = 0
        mseT = 0
        eAcumuladoT = 0
        salidasTest = []
        numPatron = []
        for i in range(nT):
            yTest = np.dot(entradasT[i], wOptimo) + bOptimo
            #Ahora calculamos el error cuadratico medio
            eFilaT = valorDeseadoT[i]-yTest
            eAcumuladoT = eAcumuladoT + (pow(eFilaT, 2))
            salidasTest.append(yTest)
            numPatron.append(i)
        mseT = ((1/nT) * eAcumuladoT)
        return mseT, salidasTest, numPatron



if __name__ == '__main__':
    # leemos y guardamos los datos
    df = pd.read_csv('dataset.csv', header=0)

    # guardamos el valor máximo y minimo de la última columna para la posterior desnormalizacion de las salidas
    max = df['ConcreteCompressiveStrength'].max()
    min = df['ConcreteCompressiveStrength'].min()

    # normalizamos los datos con el método Min-max de cada columna del dataFrame
    for col in df:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # aleatorizamos los datos
    #np.random.seed(2)
    dfRandom = df.sample(frac=1)
    train, validate, test = np.split(dfRandom, [int(.7 * len(df)), int(.85 * len(df))])

    # creamos los archivos correspondientes
    train.to_csv("train.csv", index=False)
    validate.to_csv("validate.csv", index=False)
    test.to_csv("test.csv", index=False)

    #convertimos los datos en una matriz
    dataE = train.to_numpy()
    dataV = validate.to_numpy()
    dataT = test.to_numpy()

    # factor de aprendizaje
    factAprendizaje = 0.01

    #variables de entrada de los datos
    entradasE = dataE[:, 0:8]
    entradasV = dataV[:, 0:8]
    entradasT = dataT[:, 0:8]

    #variable deseada
    valorDeseadoE = dataE[:, 8]
    valorDeseadoV = dataV[:, 8]
    valorDeseadoT = dataT[:, 8]

    #vector pesos w
    #np.random.seed(2)
    pesos = np.random.rand(8)

    #numero de muestras
    nE = len(valorDeseadoE) #721
    nV = len(valorDeseadoV)#154
    nT = len(valorDeseadoT)#155

    #bias
    b = random.uniform(0, 1)

    #ECM
    eCMV = []
    eCME = []

    #error validacion máximo
    evMinimo = 1

    #parámetro óptimos
    cicloOptimo = 0
    pesosOptimos = []
    biasOptimo = 0
    ciclos = 145

    #numero de ciclos
    numCiclo = []

    #ciclo principal
    for i in range(ciclos):
        wi, bi, mseE = entrenamiento(pesos, b)
        eCME.append(mseE)
        mseV = validacion(wi, bi)
        eCMV.append(mseV)
        numCiclo.append(i)
        #calculamos el ciclo óptimo
        if mseV < evMinimo:
            evMinimo = mseV
            cicloOptimo = i
            pesosOptimos = wi
            biasOptimo = bi

   #error de test
    eCMT, salidasTest, numPatron = fTest(pesosOptimos, biasOptimo)

    # creamos fichero de salidas con los pesos y bias optimos
    with open('valoresOptimos.txt', 'w') as valoresOptimos:
        valoresOptimos.write("Los pesos optimos son: " + str(pesosOptimos) + "\n\nEl bias optimo es: " + str(biasOptimo))


    #creamos un dataframe para ver los diferentes errores de entrenamiento y validacion en cada ciclo
    dfErrores = pd.DataFrame(list(zip(numCiclo, eCME, eCMV)), columns=['numCiclos', 'errorEntrenamiento', 'errorValidacion'])
    dfErrores.to_csv("Error_train_valid", index=False)


    #desnormalizamos los datos, y para ello primero creamos un csv de las salidas deseadas y obtenidas.
    desNorm = pd.DataFrame(list(zip(valorDeseadoT, salidasTest)), columns=['valorDeseadoTest', 'valorObtenidoTest'])

    for col in desNorm:
        desNorm[col] = desNorm[col] * (max - min) + min

    with open('deseadasVsObtenidas.txt', 'w') as output:
        for item in numPatron:
                output.write("Salida deseada para el patron " + str(
                    item) + ": " + str(desNorm['valorDeseadoTest'][item]) + "\nSalida obtenida para el patron " + str(
                    item) + ": " + str(desNorm['valorObtenidoTest'][item]) + "\n \n")

    # creamos el fichero con las salidas de la red
    with open('salidasTest.txt', 'w') as yTest:
        for item in numPatron:
            yTest.write("Patron " + str(item) + ": " + str(entradasT[item]) + "\nSalida deseada para el patron " + str(
                item) + ": " + str(valorDeseadoT[item]) + "\nSalida obtenida para el patron " + str(
                item) + ": " + str(salidasTest[item]) + "\n \n")

    #imprimimos los valores
    print(df)
    print("El nº de ciclos óptimos es: " + str(cicloOptimo))
    print("Los pesos óptimos son: " + str(pesosOptimos))
    print("El bias óptimo es: " + str(biasOptimo))
    print("El errorV minimo es: " + str(evMinimo))
    print("El error de test es: " + str(eCMT))

    # gráfica

    plt.plot(eCME, color='navy', marker='.', linewidth=2.2, markersize=0, label='ECME')
    plt.plot(eCMV, color='crimson', marker='.', linewidth=2.2, markersize=0, label='ECMV')
    plt.legend()
    plt.xlabel("nº de ciclos")
    plt.ylabel("Error cuadrático medio")
    plt.title("ECM Entrenamiento vs ECM Validación")
    plt.show()













