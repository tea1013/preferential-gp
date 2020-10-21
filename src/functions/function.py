from abc import ABCMeta, abstractmethod
import numpy as np
from scipydirect import minimize

class Function(object):
    __metaclass__ = ABCMeta

    def __init__(self, dim, bounds, noise_var, seed=1):
        self.dim = dim
        self.bounds = bounds
        self.noise_std = np.sqrt(noise_var)
        s = []
        e = []
        for bound in bounds:
            s.append(bound[0])
            e.append(bound[1])
        s = np.array(s)
        e = np.array(e)
        self.max_dist = np.linalg.norm(e - s, ord=2)
        np.random.seed(seed)


    @abstractmethod
    def value(self, x):
        pass

    @abstractmethod
    def get_pool(self, K):
        pass

    def eps(self):
        epss = np.array([])
        for bound in self.bounds:
            epss = np.append(epss, bound[1] - bound[0])

        return np.max(epss) / 1e2

    def compare(self, r, s):
        fr = -self.value(r)
        fs = -self.value(s)
        if fr >= fs:
            return 1
        else:
            return 0

        # yr = -self.value(r) + np.random.normal(0, self.noise_std)
        # ys = -self.value(s) + np.random.normal(0, self.noise_std)
        # if yr >= ys:
        #     return 1
        # else:
        #     return 0


    def ranking(self, xs):
        fs = []
        for x in xs:
            fs.append(self.value(x))

        fs = np.array(fs)

        return np.argsort(np.argsort(fs))

    def create_duels_for_db(self, K, n_duel):
        duels = []
        d1 = np.random.randint(0, K, n_duel)
        d2 = np.random.randint(0, K, n_duel)
        for i in range(n_duel):
            duels.append([d1[i], d2[i]])

        return duels

    def create_duels_for_dts(self, n_duel):
        X = []
        y = []
        for _ in range(n_duel):
            r = []
            s = []
            for d in range(self.dim):
                r.append(np.random.uniform(self.bounds[d][0], self.bounds[d][1], 1)[0])
                s.append(np.random.uniform(self.bounds[d][0], self.bounds[d][1], 1)[0])

            X.append(r + s)

            if self.compare(np.array(r), np.array(s)) == 1:
                y.append(1)
            else:
                y.append(0)

        return X, y

    def create_duels_for_pgp(self, n_duel, select_points=None):
        if self.dim == 1:
            X = np.zeros(2*n_duel)
        else:
            X = np.array([np.zeros(self.dim) for _ in range(2*n_duel)])
        y = np.zeros(2*n_duel)
        winner = np.zeros(n_duel, dtype=int)
        loser = np.zeros(n_duel, dtype=int)

        if not select_points is None:
            n_duel_ = int(len(select_points)/2)
        else:
            n_duel_ = n_duel

        for i in range(n_duel):
            if (i+1) >= (n_duel - n_duel_):
                r = np.zeros(self.dim)
                s = np.zeros(self.dim)
                for d in range(self.dim):
                    r[d] = np.random.uniform(self.bounds[d][0], self.bounds[d][1], 1)[0]
                    s[d] = np.random.uniform(self.bounds[d][0], self.bounds[d][1], 1)[0]

                if self.dim == 1:
                    r = r[0]
                    s = s[0]
            else:
                r = select_points[2*i]
                s = select_points[2*i+1]

            X[2*i] = r
            X[2*i+1] = s

            if self.compare(r, s) == 1:
                winner[i] = 2*i
                loser[i] = 2*i+1
                y[2*i] = 1
                y[2*i+1] = 0
            else:
                winner[i] = 2*i+1
                loser[i] = 2*i
                y[2*i] = 0
                y[2*i+1] = 1

        return X, y, winner, loser

    def create_from_selects(self, n_duel, select_points):
        if self.dim == 1:
            X = np.zeros(2*n_duel)
        else:
            X = np.array([np.zeros(self.dim) for _ in range(2*n_duel)])
        y = np.zeros(2*n_duel)
        winner = np.zeros(n_duel, dtype=int)
        loser = np.zeros(n_duel, dtype=int)

        for i in range(n_duel):
            r = select_points[2*i]
            s = select_points[2*i+1]
            X[2*i] = r
            X[2*i+1] = s

            if self.compare(r, s) == 1:
                winner[i] = 2*i
                loser[i] = 2*i+1
                y[2*i] = 1
                y[2*i+1] = 0
            else:
                winner[i] = 2*i+1
                loser[i] = 2*i
                y[2*i] = 0
                y[2*i+1] = 1

        return X, y, winner, loser

    def random_sampling(self, n_batch=1):
        duels = []
        for _ in range(n_batch):
            r = []
            s = []
            for d in range(self.dim):
                r.append(np.random.uniform(self.bounds[d][0], self.bounds[d][1], 1)[0])
                s.append(np.random.uniform(self.bounds[d][0], self.bounds[d][1], 1)[0])

            if self.dim == 1:
                r = r[0]
                s = s[0]

            duels.append([r, s])

        return duels

    def regret(self, x):
        return self.y_opt - (-self.value(x))
