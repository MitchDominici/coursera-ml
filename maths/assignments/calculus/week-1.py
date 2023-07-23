import numpy as np
import pandas as pd


df = pd.read_csv('../../../data/prices.csv')
# View the data with a standard print function:
# print(df)
# To print a list of the column names use columns attribute of the DataFrame:
# print(df.columns)


### START CODE HERE ### (~ 4 lines of code)
prices_A = np.array(df.price_supplier_a_dollars_per_item, dtype=np.dtype(float))
prices_B = np.array(df.price_supplier_b_dollars_per_item, dtype=np.dtype(float))
# prices_A = None(None).astype('float32')
# prices_B = None(None).astype('float32')
### END CODE HERE ###

def f_of_omega(omega):
    ### START CODE HERE ### (~ 1 line of code)
    f = (prices_A * omega) + (prices_B * (1 - omega))
    ### END CODE HERE ###
    return f

def L_of_omega(omega):
    return 1/len(f_of_omega(omega)) * np.sum((f_of_omega(omega) - np.mean(f_of_omega(omega)))**2)

# print("L(omega = 0) =",L_of_omega(0))
# print("L(omega = 0.2) =",L_of_omega(0.2))
# print("L(omega = 0.8) =",L_of_omega(0.8))
# print("L(omega = 1) =",L_of_omega(1))

# Parameter endpoint=True will allow ending point 1 to be included in the array.
# This is why it is better to take N = 1001, not N = 1000
N = 1001
omega_array = np.linspace(0, 1, N, endpoint=True)


# This is organised as a function only for grading purposes.
def L_of_omega_array(omega_array):
    N = len(omega_array)
    L_array = np.zeros(N)

    for i in range(N):
        ### START CODE HERE ### (~ 2 lines of code)
        L = L_of_omega(omega_array[i])
        L_array = L_array.at[i].set(L)
        ### END CODE HERE ###

    return L_array


L_array = L_of_omega_array(omega_array)


i_opt = L_array.argmin()
omega_opt = omega_array[i_opt]
L_opt = L_array[i_opt]
print(f'omega_min = {omega_opt:.3f}\nL_of_omega_min = {L_opt:.7f}')


# This is organised as a function only for grading purposes.
# This is organised as a function only for grading purposes.
def dLdOmega_of_omega_array(omega_array):
    N = len(omega_array)
    dLdOmega_array = np.zeros(N)

    for i in range(N):
        ### START CODE HERE ### (~ 2 lines of code)
        dLdOmega = None(None)(None[None])
        dLdOmega_array = dLdOmega_array.at[i].set(dLdOmega)
        ### END CODE HERE ###

    return dLdOmega_array


dLdOmega_array = dLdOmega_of_omega_array(omega_array)

