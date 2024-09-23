import pandas as pd
import numpy as np

# Parameters
p = 13.97e-3
s = 0.66e-3
g = 1.23e-3
w_l = 2*s
w_c = g
d = 12.935e-3
#13.97867311  0.66787077 12.93122679  1.23624968
freq = (3* 10**8)/(4*d)

# Function Definition
pi = np.pi
lamda = 4*d
theta = 0
A_p = (1/(1+ (2*p*np.sin(theta)/lamda) - (p*np.cos(theta)/(lamda))**2)**0.5) - 1
A_n = (1/(1-(2*p*np.sin(theta)/lamda) - (p*np.cos(theta)/(lamda))**2)**0.5) - 1
beta_l = np.sin(pi*w_l)/(2*p)
beta_c = np.sin(pi*w_c)/(2*p)
G_l = (0.5*((1-beta_l**2))**2*((1-beta_l**2/4)*(A_p + A_n)+ 4*(beta_l)**2*A_p*A_n))/((1-beta_l**2/4)+ beta_l**2*(1+beta_l**2/2 - beta_l**4/8)*(A_p + A_n) + 2*beta_l**6*A_p*A_n)
G_c = (0.5*((1-beta_c**2))**2*((1-beta_c**2/4)*(A_p + A_n)+ 4*(beta_c)**2*A_p*A_n))/((1-beta_c**2/4)+ beta_c**2*(1+beta_c**2/2 - beta_c**4/8)*(A_p + A_n) + 2*beta_c**6*A_p*A_n)
Xl_norm = (p/lamda)*(np.cos(theta)*(np.log(1/(np.sin(pi*w_l/(2*p))) + G_l)))
Bc_norm = 4*(d/p)*(p/lamda)*(np.cos(theta)*(np.log(1/(np.sin(pi*w_c/(2*p))) + G_c)))
Zl = (d/p)* Xl_norm

print(f"Resonant Frequency is {r_freq}Hz")
print(f"L = {L}")
print(f"C = {C}")
#r_freq = 20e9
Z = (2*pi*r_freq*L) + (1/(2*pi*r_freq*C))
print(f"Impedance of Network is {Z}")
print(f"Admittance of Network is {1/Z}")

# ABCD to S Parameter Conversion
A = 1
B = 0
C = 1/Z
D = 1
ABCD_matrix = np.array([[A,B],
                        [C,D]])
print(f"The ABCD Matrix is\n {ABCD_matrix} \n")
Zo = 377
detr = A+(B/Zo)+(C*Zo)+D

S11 = (A+(B/Zo)-(C*Zo)-D)/(detr)
S12 = 2*(A*D - B*C)/(detr)
S21 = 2/(detr)
S22 = (-A+(B/Zo)-(C*Zo)+D)/(detr)
S_matrix = np.array([[S11,S12],
                     [S21,S22]])

print(f"The S Matrix is:\n { S_matrix}\n")


print(f" S11 = {-20*np.log10(np.absolute(S11))}")
print(f" S12 = {-20*np.log10(np.absolute(S12))}")
print(f" S21 = {-20*np.log10(np.absolute(S21))}")
print(f" S22 = {-20*np.log10(np.absolute(S22))}")

import numpy as np
import pandas as pd



theta = 0
pi = np.pi


#p_range = np.arange(0.2, 51, 0.5)

s_range = np.arange(0.1e-3, 10e-3, 0.025e-3)
d_range = np.arange(0.5e-3, 15e-3,0.05e-3 )
g_range = np.arange(0.12e-3, 10e-3, 0.025e-3)


dataset = []



for s in s_range:
  for d in d_range:
    if 2*s < d:
      for g in g_range:
        lamda = 4*d
        freq = 3e8/lamda
        w_l = 2*s
        p = d + g
        w_c = g
        A_p = (1/(1+ (2*p*np.sin(theta)/lamda) - (p*np.cos(theta)/(lamda))**2)**0.5) - 1
        A_n = (1/(1-(2*p*np.sin(theta)/lamda) - (p*np.cos(theta)/(lamda))**2)**0.5) - 1
        beta_l = np.sin(pi*w_l)/(2*p)
        beta_c = np.sin(pi*w_c)/(2*p)
        G_l = (0.5*((1-beta_l**2))**2*((1-beta_l**2/4)*(A_p + A_n)+ 4*(beta_l)**2*A_p*A_n))/((1-beta_l**2/4)+ beta_l**2*(1+beta_l**2/2 - beta_l**4/8)*(A_p + A_n) + 2*beta_l**6*A_p*A_n)
        G_c = (0.5*((1-beta_c**2))**2*((1-beta_c**2/4)*(A_p + A_n)+ 4*(beta_c)**2*A_p*A_n))/((1-beta_c**2/4)+ beta_c**2*(1+beta_c**2/2 - beta_c**4/8)*(A_p + A_n) + 2*beta_c**6*A_p*A_n)
        Xl_norm = (p/lamda)*(np.cos(theta)*(np.log(1/(np.sin(pi*w_l/(2*p))) + G_l)))
        Bc_norm = 4*(d/p)*(p/lamda)*(np.cos(theta)*(np.log(1/(np.sin(pi*w_c/(2*p))) + G_c)))
        Zl = (d/p)* Xl_norm


        L = Zl/(2*pi*freq*377)
        c = Bc_norm*377/(2*pi*freq)

        r_freq = 1/(2*pi*(L*c)**0.5)
        Z = (2*pi*r_freq*L) + (1/(2*pi*r_freq*c))
        A = 1
        B = 0
        C = 1/Z
        D = 1

        Zo = 377
        detr = A+(B/Zo)+(C*Zo)+D

        S11 = (A+(B/Zo)-(C*Zo)-D)/(detr)
        S12 = 2*(A*D - B*C)/(detr)
        S21 = 2/(detr)
        S22 = (-A+(B/Zo)-(C*Zo)+D)/(detr)

        S11 = -20*np.log10(np.absolute(S11))
        S12 = -20*np.log10(np.absolute(S12))
        S21 = -20*np.log10(np.absolute(S21))
        S22 = -20*np.log10(np.absolute(S22))


        dataset.append([p, s, d, g, r_freq,S11,S21,L,c])


df2 = pd.DataFrame(dataset, columns=['p', 's', 'd', 'g', 'r_freq','S11(dB)','S21(dB)','L','C'])

print(df2)

df_cleaned = df2.dropna(subset=['r_freq'])

