"""
We consider a MFG problem with the following dynamics:
x'_1 = alpha
x'_2 = x_1
The discrete equation is:
x_1[k+1] = x_1[k] + dt * alpha[k]
x_2[k+1] = x_2[k] + dt * x_1[k]
"""


import numpy as np
import math
from origin_centered_grid import ZGrid
import functools


# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata


def listToTuple(function):
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result
    wrapper.cache_info = function.cache_info
    return wrapper


class Problem():
    """
    Coefficients and parameters of the model
    """

    def __init__(self, l, sub_l, C_l, L_l, g, L_g, A, L_A, B, L_B, C_B, T, N, dx, eps, C0, p, M0):
        self.l = l
        self.sub_l = sub_l
        self.C_l = C_l
        self.L_l = L_l
        self.g = g
        self.L_g = L_g
        self.A = A
        self.L_A = L_A
        self.B = B
        self.L_B = L_B
        self.C_B = C_B
        self.T = T
        self.N = N  # T = N * dt
        self.dx = dx
        self.dt = T / N
        self.eps = eps
        self.exp_p = p
        q = p / (p - 1)  # conjugate of p
        C_p_aux = (2 / sub_l) * (2 * T * C_l + (T * (L_g ** q) * math.exp(q * T * L_A) * (C_B ** q)) / (
                    ((sub_l * p / 2) ** (q / p)) * q))
        L_v = (L_l * (T + C_p_aux) + L_g) * math.exp(T * L_A + L_B * (T ** (1 / q)) * (C_p_aux ** (1 / p)))
        alpha_aux = (2 / sub_l) * (2 * C_l + ((L_v ** q) * (C_B ** q)) / (((sub_l * p / 2) ** (q / p)) * q))
        #Trunco el valor de C_alpha para disminuir el tamanio de la grilla
        self.C_alpha = min(0.4,alpha_aux ** (1 / p))
        C = np.zeros(N + 1)
        C[0] = C0
        for i in range(N):
            C[i + 1] = C[i] + self.dt * max(C[i], self.C_alpha) + self.dx

        self.C_list = C  # list of bounds for the compact sets
        #N_list is the list of radiuos of the grids of the sequence of compact sets
        N_list = []
        for k in range(self.N + 1):
            N_list.append(int(self.C_list[k] / self.dx))
        self.N_list = N_list
        self.M0 = M0  # M0 only depends on x.



    # Given V (which will be V[k+1]) and M (M[k]), we obtain the probability from [x_index[0], x_index[1]] to a point wiht first component y0_index.
    def compute_P_opt_aux(self, V, M):
        def p_opt_k(x_index,y0_index):
            xx = (x_index[0] * self.dx, x_index[1] * self.dx)
            # in this case only matters the first component with dynamics x_1[k+1] = x_1[k] + dt * alpha
            N_1_x = int(self.dt * self.C_alpha / self.dx)
            #print("N_1_X =", N_1_x)
            if N_1_x == 0:
                if x_index[0] == y0_index:
                    return 1
                else:
                    return 0

            if abs(x_index[0]-y0_index) > N_1_x:
                return 0
            else:
                # interpolation  x_int = gamma * x[k_int] + (1-gamma) * x[k_int+1]
                x_int = xx[1] + self.dt * xx[0]
                k_int = math.floor(x_int / self.dx)
                gamma = k_int + 1 - (x_int / self.dx)
                sump = 0
                exponents = []
                for j in range(-N_1_x , N_1_x + 1):
                    y_aux = (x_index[0] + j) * self.dx
                    #print('l=', self.l(((y_aux - xx[0]) / self.dt), (x_index[0],x_index[1]), M), 'V=',  V[x_index[0] + j, k_int], 'exp=', -(self.dt * self.l(((y_aux - xx[0]) / self.dt), (x_index[0],x_index[1]), M) + gamma * V[x_index[0] + j, k_int]
                                       #+ (1 - gamma) * V[x_index[0] + j, k_int + 1]) / self.eps)
                    exponents.append(-(self.dt * self.l(((y_aux - xx[0]) / self.dt), (x_index[0],x_index[1]), M) + gamma * V[x_index[0] + j, k_int]
                                       + (1 - gamma) * V[x_index[0] + j, k_int + 1]) / self.eps)

                min_exponent = min(exponents)
                for i in range(len(exponents)):
                    sump += math.exp(exponents[i] - min_exponent)


            return math.exp(-(self.dt * self.l(((y0_index * self.dx - xx[0]) / self.dt), (x_index[0],x_index[1]), M) + gamma * V[y0_index, k_int] + (
                            1 - gamma) * V[y0_index, k_int + 1]) / self.eps - min_exponent) / sump

        return p_opt_k

    # M is a list (M[0], M[1], ..., M[N]) of distributions of the state
    # The function compute_P_opt returns a list of function P such that P[k] is the transition probability for the first component
    def compute_P_opt(self, M):

        print('entramos a compute_P_opt')
        V = []
        # we define V(N) given by the function g(.,M(N))
        V_N = ZGrid(self.N_list[self.N])
        for i in range(-self.N_list[self.N], self.N_list[self.N] + 1):
            for j in range(-self.N_list[self.N], self.N_list[self.N] + 1):
                V_N[i, j] = self.g((i, j), M[self.N])
        V.append(V_N)
        print('Calculamos hasta V: N')

        p_opt = []
        for k in range(self.N - 1, -1, -1):  # recorro la lista backward: N-1, N-2, ....,0
            V_k = ZGrid(self.N_list[k])
            p_opt_k = self.compute_P_opt_aux(V[0], M[k])  # for k, V = [V(k+1),..., V(N)]
            for i in range(-self.N_list[k], self.N_list[k] + 1):
                for j in range(-self.N_list[k], self.N_list[k] + 1):
                    xx = (i * self.dx, j * self.dx)
                    # interpolation
                    x_int = xx[1] + self.dt * xx[0]
                    k_int_xx = math.floor(x_int / self.dx)
                    gamma_xx = k_int_xx + 1 - (x_int / self.dx)
                    # N_1_X_aux is the radius of the neighborhood of the point x w.r.t. the first component
                    N_1_x_aux = int(self.dt * self.C_alpha / self.dx)
                    for i_aux in range(-N_1_x_aux, N_1_x_aux + 1):
                        V_k[i, j] += p_opt_k([i, j], i + i_aux) * (self.dt * self.l((i_aux * self.dx / self.dt), (i,j), M[k]) + gamma_xx * V[0][i + i_aux, k_int_xx] + (1 - gamma_xx) * V[0][i + i_aux, k_int_xx + 1])
            V.insert(0, V_k)
            p_opt.insert(0, p_opt_k)
            print('Calculamos hasta V y p_opt:', k)
        return p_opt

    # beta function centered at y in R, evaluated in x in R
    def compute_beta(self, y, x):
        # print("compute_beta")
        if abs(x - y) > self.dx:
            return 0
        else:
            if x < y:
                return (x - y) / self.dx  + 1
            if x >= y:
                return -(x - y) / self.dx + 1

    def compute_M(self, p_opt):
        M = []
        #M0_aux = ZGrid(self.N_list[self.N])
        M0_aux = ZGrid(self.N_list[0])


        for i in range(-self.N_list[0], self.N_list[0] + 1):
            for j in range(-self.N_list[0], self.N_list[0] + 1):
                M0_aux[i,j] = self.M0((i * self.dx, j * self.dx))
        M.append(M0_aux)

        for k in range(1, self.N + 1):
            #M_k = ZGrid(self.N_list[self.N])
            M_k = ZGrid(self.N_list[k])
            for i in range(-self.N_list[k], self.N_list[k] + 1):
                for j in range(-self.N_list[k], self.N_list[k] + 1):
                    sum_aux = 0
                    for i_aux in range(-self.N_list[k-1], self.N_list[k-1] + 1):
                        for j_aux in range(-self.N_list[k-1], self.N_list[k-1] + 1):
                            sum_aux += p_opt[k-1]([i_aux, j_aux], i) * self.compute_beta(j * self.dx, j_aux * self.dx + self.dt * i_aux * self.dx) * M[k-1][i_aux, j_aux]

                    M_k[i, j] = sum_aux
            M.append(M_k)
            print('calculamos hasta M[',k,']')
        return M


    def update_M(self, n, bar_M, BR_M):
        M = []

        for k in range(self.N+1):
            M_k = ZGrid(self.N_list[k])
            for i in range(-self.N_list[k], self.N_list[k] + 1):
                for j in range(-self.N_list[k], self.N_list[k] + 1):
                    M_k[i,j] = (n / (n + 1)) * bar_M[k][i,j] + BR_M[k][i,j] / (n + 1)
            M.append(M_k)
        return M


    # Las marginales las dejamos de longitud 2 * self.N_list[self.N] + 1 para poder graficar.
    def marginal_1(self, M):
        M1 = []
        for k in range(self.N + 1):
            M1_k = np.zeros(2 * self.N_list[self.N] + 1)
            for i in range(-self.N_list[k], self.N_list[k] + 1):
                aux = 0
                for j in range(-self.N_list[k], self.N_list[k] + 1):
                    aux += M[k][i,j]
                M1_k[i + self.N_list[self.N]] = aux
            M1.append(M1_k)
        return M1

    def marginal_2(self, M):
        M2 = []
        for k in range(self.N + 1):
            M2_k = np.zeros(2 * self.N_list[self.N] + 1)
            for j in range(-self.N_list[k], self.N_list[k] + 1):
                aux = 0
                for i in range(-self.N_list[k], self.N_list[k] + 1):
                    aux += M[k][i,j]
                M2_k[j + self.N_list[self.N]] = aux
            M2.append(M2_k)
        return M2


    def norm_l1(self, M1, M2):
        aux = 0
        for k in range(self.N+1):                    #here we assume that M[k] is supported in B(C_list[k])
            for i in range(-self.N_list[k], self.N_list[k]+1):
                for j in range(-self.N_list[k], self.N_list[k] + 1):
                    aux += self.dt * abs(M1[k][i,j]-M2[k][i,j])

        return aux




