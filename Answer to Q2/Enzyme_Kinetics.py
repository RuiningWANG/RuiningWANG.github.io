from re import M
import numpy as np
import matplotlib.pyplot as plt

#define the global parameters: initial concentrations and reaction velocities
E = 1
ES = 0
P = 0
k1 = 100/60
k2 = 600/60
k3 = 150/60
h = 0.01

#define function for four substrates S,ES,P,E
def func_S(e,s,es,t):
    m = k2*es - k1*e*s
    return m

def func_ES(e,s,es,t):
    m = k1*e*s - k2*es - k3*es
    return m

def func_P(e,s,es,t):
    m = k3*es
    return m

def func_E(e,s,es,t):
    m = k2*es + k3*es - k1*e*s
    return m

#define function for the fourth-order Runge-Kutta formula
def formula(ei,si,esi,pi,ti,h):
    k1e = func_E(ei,si,esi,ti)*h
    k1s = func_S(ei,si,esi,ti)*h
    k1es = func_ES(ei,si,esi,ti)*h
    k1p = func_P(ei,si,esi,ti)*h
    
    k2e = func_E(ei+k1e/2, si+k1s/2,esi+k1es/2, ti+h/2)*h
    k2s = func_S(ei+k1e/2, si+k1s/2,esi+k1es/2, ti+h/2)*h
    k2es = func_ES(ei+k1e/2, si+k1s/2,esi+k1es/2, ti+h/2)*h
    k2p = func_P(ei+k1e/2, si+k1s/2,esi+k1es/2, ti+h/2)*h
    
    k3e = func_E(ei+k2e/2, si+k2s/2,esi+k2es/2, ti+h/2)*h
    k3s = func_S(ei+k2e/2, si+k2s/2,esi+k2es/2, ti+h/2)*h
    k3es = func_ES(ei+k2e/2, si+k2s/2,esi+k2es/2, ti+h/2)*h
    k3p = func_P(ei+k2e/2, si+k2s/2,esi+k2es/2, ti+h/2)*h
    
    k4e = func_E(ei+k3e, si+k3s,esi+k3es, ti+h)*h
    k4s = func_S(ei+k3e, si+k3s,esi+k3es, ti+h)*h
    k4es = func_ES(ei+k3e, si+k3s,esi+k3es, ti+h)*h
    k4p = func_P(ei+k3e, si+k3s,esi+k3es, ti+h)*h
    
    to = ti+h
    eo = ei + (k1e+k4e+(k2e+k3e)*2)/6
    so = si + (k1s+k4s+(k2s+k3s)*2)/6
    eso = esi + (k1es+k4es+(k2es+k3es)*2)/6
    po = pi + (k1p+k4p+(k2p+k3p)*2)/6
    
    return to,eo,so,eso,po

#the process of simulation
def process(S):
    datapoints = [] #1column:time points
                    #2-5column:concentrations of E, S, ES, P; 
                    #6column: velocity of P, i.e. rate of chnge of P
    
    t = 0
    while t<100:
        to,Eo,So,ESo,Po = formula(E,S,ES,P,t,h)
        
        # calculate the change velocity of P
        v = (Po-P)/(to-t)
        
        #record in the data matrix
        datapoints.append(np.array([to,Eo,So,ESo,Po,v]))
        
        #update number
        E,S,ES,P,t = Eo,So,ESo,Po,to
        
    return datapoints

#answer to 8.2
#derive the curve points
curve_2 = np.array(process(S=10))

#plot the curves in one picture
plt.plot(curve[:,0], curve[:,1], label="E")
plt.plot(curve[:,0], curve[:,2], label="S")
plt.plot(curve[:,0], curve[:,3], label="ES")
plt.plot(curve[:,0], curve[:,4], label="P")
plt.title("Concentrations of reactants over time")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (uM)")
plt.legend()
plt.show()

#derive the subtrate P's curve
curve_P = []
timeline = []
for S in range (1,100):
    inter_value = np.array(process(S))
    curve_value = np.max(inter_value[:,5])
    curve_P.append(curve_value)
    timeline.append(S)
    

#plot and output outcomes
Vm = np.max(curve_P)
print(f'The maximum rate of P is {Vm:.3f} uM/s.')

plt.plot(timeline, curve_P, label="P")
plt.title("Rate of change of P over concentration of S")
plt.xlabel("Concentration of S (uM)")
plt.ylabel("Rate of change of P (uM/s)")
plt.legend()
plt.show()