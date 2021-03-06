import numpy as np
import matplotlib.pyplot as plt
import math

#from MFG_matrix_auxi_truncation import Problem, listToTuple
#from MFG_matrix_auxiliary import Problem, listToTuple
from MFG_auxiliary import Problem, listToTuple
from origin_centered_grid import ZGrid
import functools

#from scipy.interpolate import griddata
#This algorithm works for the example
#M1 is an array of functions M1=[M1[0], ..., M1[N]], M[i](x1,x2) is the probability of (x1,x2) at time t_i.
# #The support of M1[i] is contained in Gi, to be defined. C0 bound of K_0 = supp(M1[0]) (m_0).


def best_response(mfg, M1):
    p_opt = mfg.compute_P_opt(M1)
    return mfg.compute_M(p_opt)

#this function show the figura for M(k,x), where k is the time t_k and x in R the grid points
def show_M(mfg,M):
    x_state = []
    for i in range(-mfg.N_list[mfg.N], mfg.N_list[mfg.N] + 1):
        x_state.append(mfg.dx * i)
    y_times = []
    for i in range(mfg.N):
        y_times.append(mfg.dt * i)
    M_values = []
    for k in range(mfg.N):
        list_aux = []
        for j in range(2 * mfg.N_list[mfg.N] + 1):
            list_aux.append(M[k][j])
        M_values.append(list_aux)
    fig, ax = plt.subplots()
    Z = np.array(M_values)
    ax.imshow(Z)
    plt.show()


# def main_alg(M1, N, dt, dx, Cal, C0, l, g, A, B, eps, tol):
def main_alg(mfg, tol):      #M1, tol):
    print("N", mfg.N)
    print("C list:", mfg.C_list)
    print("N list:", mfg.N_list)
    print("C aplpha: ", mfg.C_alpha)
    e = tol + 1
    M1 = []
    #M1[k] = M0 for all k:
    for k in range(mfg.N + 1):
        M1_k = ZGrid(mfg.N_list[k])
        for i in range(-mfg.N_list[0], mfg.N_list[0] + 1):
            for j in range(-mfg.N_list[0], mfg.N_list[0] + 1):
                M1_k[i, j] = M0((i * mfg.dx, j * mfg.dx))
        M1.append(M1_k)

    M_bar = M1
    n = 1

    print("About to enter the first while...")

    while e > tol:
        print("entro al while")
        k = 1
        e_aux = e
        while e_aux > (e - tol):
            M_aux = best_response(mfg, M_bar)
            e_aux = mfg.norm_l1(M_bar, M_aux)
            M_bar = mfg.update_M(k, M_bar, M_aux)
            k += 1
            print('e_aux:', e_aux, 'k:', k)
        e = e_aux
        n = n + k - 1

    print('n=', n)

    show_M(mfg, mfg.marginal_1(M_bar))
    show_M(mfg, mfg.marginal_2(M_bar))
    return M_bar  # o M_aux??




def A(x):
    return [0,x[0]]
def B(x):
    return [1,0]


C0 = 0.4
dx = 0.008
N = 30

#initial distribution for state and control (x[0],x[1]). The distribution of x[0] is delta_0.

@functools.lru_cache(maxsize=None)
def M0(x):
    x2_grid = []
    N_state_0 = int(C0/dx)
    for i in range(-N_state_0, N_state_0 + 1):
        x2_grid.append(dx * i)
    sum_aux_1 = 0
    sum_aux_2 = 0
    for x2 in x2_grid:
        #sum_aux_1 += math.exp(-((x2) ** 2) / 0.05)
        sum_aux_2 += math.exp(-((x2+0.1)**2)/0.01)

    if abs(x[0]) > C0 or abs(x[1]) > C0:
        return 0
    #else:
    #    return (math.exp(-(x[0])**2/0.05) * math.exp(-((x[1]+0.1)**2)/0.02)) / (sum_aux_1 * sum_aux_2)
    if x[0] == 0:
        if abs(x[1]) > C0:
            return 0
        else:
            return math.exp(-((x[1]+0.1)**2)/0.01)/sum_aux_2
    else:
        return 0



@functools.lru_cache(maxsize=None)
def double_convolution(sigma,M):
    ## en la normal en el intervalo [-2 sigma, 2 sigma ] está más del 95% de la distribución
    N_x_aux = int(sigma / dx)
    rho = np.zeros(4 * N_x_aux + 1)
    for j in range(4 * N_x_aux + 1):
        rho[j] = 0.5 * math.exp(-(((j - 2 * N_x_aux) * dx) ** 2) / (2 * (sigma ** 2))) / math.sqrt(2 * math.pi * (sigma ** 2))
    #print('max rho:', max(rho))
    radius = M.radius
    #second marginal of M: M_2
    M_2 = np.zeros(2 * radius + 1)
    for j in range(-radius, radius + 1):
        sum = 0
        for i in range(-radius, radius + 1):
            sum += M[i, j]
        M_2[j + radius] = sum
    #print('max M_2:', max(M_2))
    conv = np.convolve(rho, np.convolve(rho,
                                        M_2))  # longitud impar, ya que la conv de dos listas de N y M es N+M-1, y en nuestro casos todas las listas son impares

    in_aux = int((len(conv) - (2 * radius + 1)) / 2)
    ret = conv[in_aux: in_aux + 2 * radius + 1: 1]
    #print('ret=', ret)
    #print('max convolutions:', max(ret))

    return ret



def l(alpha, x, M):
    f_values = double_convolution(0.2,M)

    return ((alpha)**2)/20 +  (x[1] * dx - 0.3) ** 2 + f_values[x[1] + M.radius]


def g(x, M):

    f_values = double_convolution(0.2,M)

    return (x[1] * dx - 0.3)**2 / 4 + f_values[x[1] + M.radius]



if __name__ == '__main__':
    sub_l = 1/20
    C_l = 1
    L_l = 1
    L_g  = 0.25 #0.3
    L_A = 1
    L_B = 0
    C_B = 1
    T = 1
    eps = 0.005
    p = 2
    tol = 0.1

    mfg = Problem(l, sub_l, C_l, L_l, g, L_g, A, L_A, B, L_B, C_B, T, N, dx, eps, C0, p, M0)
    main_alg(mfg, tol)

    print('M0=', M0.cache_info())
    print('l=', double_convolution.cache_info())



#quizás en best response puedo devolver también a P, así calculo a V y tengo el triplete (V_n, M_n, bar M_n).
