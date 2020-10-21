import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipydirect import minimize
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 14 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 14 # 軸だけ変更されます
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid

class PreferentialGP:
    def __init__(self, f, kernel, noise_var, X, y, winner, loser):
        self.f = f
        self.kernel = kernel
        self.noise_var = noise_var
        self.X = X
        self.y = y
        self.winner = np.array(winner)
        self.loser = np.array(loser)
        self.n = len(X)
        self.m = len(winner)

        self.calc_params()

    def z(self, f):
        z = np.zeros(self.m)
        for k in range(self.m):
            v_k, u_k = self.winner[k], self.loser[k]
            z[k] = (f[v_k] - f[u_k]) / (np.sqrt(2) * np.sqrt(self.noise_var))

        return z

    def cdf_z(self, f):
        cdf_z = np.array([])
        z = self.z(f)
        for k in range(self.m):
            cdf_z = np.append(cdf_z, norm.cdf(z[k], loc=0, scale=1))

        return cdf_z

    def pdf_z(self, f):
        pdf_z = np.array([])
        z = self.z(f)
        for k in range(self.m):
            pdf_z = np.append(pdf_z, norm.pdf(z[k], loc=0, scale=1))

        return pdf_z

    def S(self, f):
        cdf_z = self.cdf_z(f)
        S = -np.sum(np.log(cdf_z)) + 0.5 * np.dot(f.T, np.dot(self.sigma_inv, f))

        return S

    def map_estimation(self):
        eps = 1e-6
        f_map = np.zeros(self.n)

        sigma_inv_sum = self.sigma_inv + self.sigma_inv.T

        while(True):
            cdf_z = self.cdf_z(f_map)
            pdf_z = self.pdf_z(f_map)

            delta_cdf = np.zeros(self.n)

            for k in range(self.m):
                v_k, u_k = self.winner[k], self.loser[k]

                delta_cdf[v_k] += (-1 / (np.sqrt(2) * np.sqrt(self.noise_var))) * (pdf_z[k] / (cdf_z[k]))
                delta_cdf[u_k] += (1 / (np.sqrt(2) * np.sqrt(self.noise_var))) * (pdf_z[k] / (cdf_z[k]))

            delta_1 = np.zeros(self.n)
            delta_1 += delta_cdf
            delta_1 += 0.5 * np.dot(sigma_inv_sum, f_map)

            delta_2 = np.zeros(shape=(self.n, self.n))
            delta_2 += self.calc_lambda(f_map)
            delta_2 += self.sigma_inv

            f_map_before = np.copy(f_map)
            f_map = f_map - np.dot(np.linalg.inv(delta_2), delta_1)

            if np.all(np.sum((f_map - f_map_before)**2) < eps):
                break

            # if np.all(np.sum((f_map - f_map_before)) < eps):
            #     break

        return f_map

    def calc_beta(self, f_map):
        beta = np.zeros(self.n)
        cdf_z = self.cdf_z(f_map)
        pdf_z = self.pdf_z(f_map)
        for k in range(self.m):
            v_k, u_k = self.winner[k], self.loser[k]

            beta[v_k] += (1 / (np.sqrt(2) * np.sqrt(self.noise_var))) * (pdf_z[k] / cdf_z[k])
            beta[u_k] += (-1 / (np.sqrt(2) * np.sqrt(self.noise_var))) * (pdf_z[k] / cdf_z[k])

        return beta

    def calc_lambda(self, f_map):
        Lambda = np.zeros(self.sigma.shape)

        z = self.z(f_map)
        cdf_z = self.cdf_z(f_map)
        pdf_z = self.pdf_z(f_map)

        for k in range(self.m):
            v_k, u_k = self.winner[k], self.loser[k]
            p = (-1 / (2 * self.noise_var)) * ((pdf_z[k]**2 / cdf_z[k]**2) + (z[k] * pdf_z[k] / cdf_z[k]))
            q = (1 / (2 * self.noise_var)) * ((pdf_z[k]**2 / cdf_z[k]**2) + (z[k] * pdf_z[k] / cdf_z[k]))
            Lambda[v_k][u_k] += p
            Lambda[u_k][v_k] += p
            Lambda[v_k][v_k] += q
            Lambda[u_k][u_k] += q

        Lambda += np.eye(len(f_map)) * 1e-6

        return Lambda

    def predict(self, duel):
        if self.f.dim == 1:
            k_t = self.kernel(np.atleast_2d(duel).T, np.atleast_2d(self.X).T).T
            sigma_t = self.kernel(np.atleast_2d(duel).T)
        else:
            d = duel.reshape(-1, self.f.dim)
            k_t = self.kernel(d, self.X).T
            sigma_t = self.kernel(d)

        mu_star = np.dot(k_t.T, self.beta)
        sigma_star = sigma_t - k_t.T.dot(self.sigma_lambda_map_inv_sum).dot(k_t)

        return mu_star, sigma_star

    def p(self, duel):
        if (duel[0:self.f.dim] == duel[self.f.dim:2*self.f.dim]).all():
            return 0.5

        mu_star, sigma_star = self.predict(duel)

        mu_star_r = mu_star[0]
        mu_star_s = mu_star[1]

        sigma_star_rr = sigma_star[0][0]
        sigma_star_rs = sigma_star[0][1]
        sigma_star_sr = sigma_star[1][0]
        sigma_star_ss = sigma_star[1][1]

        variance_star = sigma_star_rr + sigma_star_ss - sigma_star_rs - sigma_star_sr
        std_star = np.sqrt(variance_star)
        p = norm.cdf((mu_star_r - mu_star_s) / std_star, loc=0, scale=1)

        return p

    def mean_var(self, xs):
        if self.f.dim == 1:
            xs = np.atleast_2d(xs)
            k_t = self.kernel(xs.T, np.atleast_2d(self.X).T).T
        else:
            k_t = self.kernel(xs.reshape(-1, self.f.dim), self.X).T

        mu_star = np.dot(k_t.T, self.beta)

        sigma_t = np.ones(len(xs))
        sigma_star = sigma_t - np.einsum("ij,jk,ki->i", k_t.T, self.sigma_lambda_map_inv_sum, k_t)

        return mu_star, sigma_star

    def mu(self, x):
        if self.f.dim == 1:
            x = np.atleast_2d(x)
            k_t = self.kernel(x.T, np.atleast_2d(self.X).T)
        else:
            x = np.atleast_2d(x)
            k_t = self.kernel(x, self.X)

        mu_star = np.dot(k_t, self.beta)

        return mu_star

    def mu_minus(self, x):
        return -self.mu(x)

    def calc_params(self):
        if self.f.dim == 1:
            self.sigma = self.kernel(np.atleast_2d(self.X).T)
        else:
            self.sigma = self.kernel(self.X)

        self.sigma += np.eye(self.n) * self.noise_var
        self.sigma_inv = np.linalg.inv(self.sigma)

        f_map = self.map_estimation()

        beta = self.calc_beta(f_map)
        lambda_map = self.calc_lambda(f_map)

        self.f_map = f_map
        self.beta = beta
        self.lambda_map = lambda_map
        self.lambda_map_inv = np.linalg.inv(self.lambda_map)
        self.sigma_lambda_map_inv_sum = np.linalg.inv(self.sigma + self.lambda_map_inv)
        self.sigma_inv_lambda_map_sum_inv = np.linalg.inv(self.sigma_inv + self.lambda_map)

    def marginal_liklihood(self, lengthscale):
        if lengthscale == 0:
            return 1
        self.kernel.set(lengthscale)
        self.calc_params()
        marginal_liklihood = np.exp(-self.S(self.f_map)) * np.linalg.det(np.eye(self.n) + self.sigma.dot(self.lambda_map))**(-0.5)

        return -marginal_liklihood

    def fit(self):
        bound = self.f.lengthscale_bound
        res = minimize(self.marginal_liklihood, bound, maxf=1000, algmethod=1)
        lengthscale = res['x'][0]
        self.kernel.lengthscale = lengthscale

    def opt_x(self):
        res = minimize(self.mu_minus, self.f.bounds, maxf=self.f.dim * 1000, algmethod=1)
        return res['x']

    def samplng_posterior(self):
        f = np.random.multivariate_normal(self.f_map, self.sigma_inv_lambda_map_sum_inv)
        return f

    def plot_predict(self, dir_path, file_name, opt_x=None, next_duel=None, label=None, scatter=True, savefig=True):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        xs = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 1000)
        mu, var = self.mean_var(xs)
        std = np.sqrt(var)
        plt.plot(xs, mu, 'g', zorder=1)

        if self.f.dim == 1:
            if scatter:
                for i in range(int(len(self.X)/2)):
                    x1, x2 = self.X[2*i], self.X[2*i+1]
                    y1, y2 = self.y[2*i], self.y[2*i+1]
                    mu1, _ = self.mean_var(x1)
                    mu2, _ = self.mean_var(x2)
                    if y1 > y2:
                        plt.scatter(x1, mu1, c='r', zorder=3)
                        plt.scatter(x2, mu2, c='b', zorder=3)
                    else:
                        plt.scatter(x1, mu1, c='b', zorder=3)
                        plt.scatter(x2, mu2, c='r', zorder=3)

                if not next_duel is None:
                    x1, x2 = next_duel[0], next_duel[1]
                    mu1, _ = self.mean_var(x1)
                    mu2, _ = self.mean_var(x2)
                    plt.scatter(x1, mu1, c='black', s=80, marker='*', zorder=3)
                    plt.scatter(x2, mu2, c='black', s=80, marker='*', zorder=3)

            if not opt_x is None:
                for x in opt_x:
                    plt.axvline(x=x, zorder=2)

            plt.fill_between(xs, mu - std, mu + std, color='g', alpha=0.3)
            plt.xlabel('x')
            plt.ylabel('f')
            if not label is None:
                plt.title(label)

            if savefig:
                plt.savefig('{}/{}'.format(dir_path, file_name))
                plt.close()
            else:
                plt.show()
