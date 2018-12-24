#Thermo Project
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import fsolve
from scipy.optimize import brentq

Psat = dict()
Psat['3-ethyltoluene'] = lambda T: math.exp(13.7819 - 2726.81/(217.572 + T))
#3-ethyltoluene: https://webbook.nist.gov/cgi/cbook.cgi?ID=C620144&Units=SI&Mask=7FF
Psat['isooctane'] = lambda T: math.exp(13.6703 - 2896.31/(220.767 + T))
#isooctane: https://webbook.nist.gov/cgi/cbook.cgi?ID=C540841&Mask=FFF
Psat['ethanol'] = lambda T: math.exp(16.8958 - 3795.17 /(230.918 + T))
#ethanol: https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Type=ANTOINE&Plot=on#ANTOINE
Psat['butane'] = lambda T: math.exp(13.6608 - 2154.70/(238.789 + T))
#butane: https://webbook.nist.gov/cgi/cbook.cgi?ID=C106978&Mask=4&Type=ANTOINE&Plot=on

z = dict()

species = ['ethanol', 'isooctane', '3-ethyltoluene', 'butane']
T = 55 
#x = z
z['ethanol'] = 0.1
z['isooctane'] = 0.3
z['3-ethyltoluene'] = 0.3
z['butane'] = 1 - sum(z.values())

def Bubble_Point(species, T):
    return sum([z[s]*(Psat[s](T)) for s in species])

def ybub(species,x):
    return {s: z[s]*Psat[s](T)/Bubble_Point(species, T) for s in species}

P = 191 #~Bubble_Point(species,T)

K = {n : lambda P,T,n=n: Psat[n](T)/P for n in Psat}

print("Pressure     {:6.2f} [mmHg]".format(P))
print("Temperature  {:6.2f} [deg C]".format(T))
print("K-factors:")
for n in K:
    print("   {:s}  {:7.3f}".format(n,K[n](P,T)))

def RR(phi):
    return sum([(K[n](P,T)-1)*z[n]/(1 + phi*(K[n](P,T)-1)) for n in K.keys()])

phi = np.linspace(0,1)
plt.plot(phi,[RR(phi) for phi in phi])
plt.xlabel('Vapor Fraction phi')
plt.title('Rachford-Rice Equation')
plt.grid()
plt.show()
 
phi = brentq(RR,0,1)

print("Vapor Fraction  {:6.4f}".format(phi))
print("Liquid Fraction {:6.4f}".format(1-phi))

x = {n: z[n]/(1 + phi*(K[n](P,T)-1)) for n in z}
y = {n: K[n](P,T)*z[n]/(1 + phi*(K[n](P,T)-1)) for n in z}

print("Component    z[n]    x[n]    y[n]")

liquid_ratio = phi
vapor_ratio = 1-phi

for n in z.keys():
    print("{:10s} {:6.4f}  {:6.4f}  {:6.4f}".format(n,z[n],x[n],y[n]))

def filter_list_liquid(x):
    totalSales = x[3]
    FuelHeight = x[28]
    Temperature_Tank = x[29]
    liquid_Volume = x[26]
    return [totalSales, Temperature_Tank, liquid_Volume, liquid_ratio, FuelHeight]

def filter_list_vapor(x):
    totalSales = x[3]
    Vapor_Temp = x[31]
    vapor_volume = x[35]
    return [totalSales, Vapor_Temp, vapor_ratio, vapor_volume]

with open('Thermo_Data_Training.csv') as csvfile:
    data = csv.reader(csvfile)
    result_t_liquid = []
    result_t_vapor = []
    row_i = 0
    for row in data:
        if row_i == 0: 
            row_i += 1
            continue
        result_t_liquid.append(filter_list_liquid(list(row)))
        result_t_vapor.append(filter_list_vapor(list(row)))
        row_i += 1
    arr_liquid_t_x = np.array(result_t_liquid, dtype='f')
    arr_vapor_t_x = np.array(result_t_vapor, dtype='f')

with open('Thermo_Data_Test.csv') as csvfile:
    data = csv.reader(csvfile)
    result_liquid = []
    result_vapor = []
    row_i = 0
    for row in data:
        if row_i == 0: 
            row_i += 1
            continue
        result_liquid.append(filter_list_liquid(list(row)))
        result_vapor.append(filter_list_vapor(list(row)))
        row_i += 1
    arr_liquid_x = np.array(result_liquid, dtype='f')
    arr_vapor_x = np.array(result_vapor, dtype='f')

arr_l_t_x = arr_liquid_t_x[:, :-1]
arr_l_t_y = arr_liquid_t_x[:, -1]
arr_v_t_x = arr_vapor_t_x[:, :-1]
arr_v_t_y = arr_vapor_t_x[:, -1]
arr_l_x = arr_liquid_x[:, :-1]
arr_l_y = arr_liquid_x[:, -1]
arr_v_x = arr_vapor_x[:, :-1]
arr_v_y = arr_vapor_x[:, -1]

liquid_model = linear_model.LinearRegression(normalize=True)
liquid_model.fit(arr_l_t_x, arr_l_t_y)
print "Liquid Model Score:", abs(liquid_model.score(arr_l_x, arr_l_y))

vapor_model = linear_model.LinearRegression(normalize=True)
vapor_model.fit(arr_v_t_x,arr_v_t_y)
print "Vapor Model Score:", abs(vapor_model.score(arr_v_x, arr_v_y))