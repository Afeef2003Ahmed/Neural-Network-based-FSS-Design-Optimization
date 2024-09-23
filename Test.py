frequencies = 1
s11 = 8.34  #1.714331   8.345105  100.347692
s21 = 100.34
normalized_frequencies =(frequencies - mean['r_freq']) / std['r_freq']
s11_norm = (s11 - mean['S11(dB)']) / std['S11(dB)']
s21_norm = (s21 - mean['S21(dB)']) / std['S21(dB)']
Norm_input = np.array([[normalized_frequencies,s11_norm,s21_norm]])
predictions = model.predict(Norm_input)

print(Norm_input)
print(predictions[0])

Train_std = std[['p', 's', 'd','g']].to_numpy()
Train_mean = mean[['p', 's', 'd','g']].to_numpy()
Prediction = predictions * Train_std + Train_mean
print(Prediction)

p = Prediction[0][0]*(1e-3)
s = Prediction[0][1]*(1e-3)
d = Prediction[0][2]*(1e-3)
g = Prediction[0][3]*(1e-3)
w_l = 2*s
w_c = g
freq = (3* 10**8)/(4*d)

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


L = Zl/(2*pi*freq*377)
C = Bc_norm*377/(2*pi*freq)

r_freq = 1/(2*pi*(L*C)**0.5)

print(f"Resonance frequency is {r_freq}")
print(f"Input Frequency is {frequencies*(1e10)}")
print(f"The Inductance value is {L}")
print(f"The Capacitance value is {C}")
#print(f"Frequency Aprrox is {freq}")

test_predictions = model.predict(val_features)

# Calculate Mean Squared Error (MSE) for the test set
mse = np.mean((test_predictions - val_labels) ** 2)

# Calculate R-squared (R2) for the test set
ssr = np.sum((test_predictions - val_labels) ** 2)  # Sum of squared residuals
sst = np.sum((val_labels - np.mean(val_labels)) ** 2)  # Total sum of squares
r2 = 1 - (ssr / sst)

print("Mean Squared Error on Test Set: {:.3f}".format(mse))
print("R-squared on Test Set: {:.3f}".format(r2))




