import numpy as np


def get_pq(df):
    p = df[df['model_index'] == 0]
    p = p.set_index(['note_index']).value

    q = df[df['model_index'] == 1]
    q = q.set_index(['note_index']).value
    return p, q


def get_group_div(fun, sqrt):
    def kl_group(df):
        p, q = get_pq(df)
        if sqrt:
            return np.sqrt(fun(p, q))
        else:
            return fun(p, q)

    return kl_group


def KL(p, q):
    eps = 1e-10
    return (p * np.log((p + eps) / (q + eps))).sum()


def JS(p, q):
    m = (p + q) / 2
    return (KL(p, m) + KL(q, m))


def reversed_KL(p, q):
    return KL(q, p)


def Jeffreys(p, q):
    return KL(p, q) + KL(q, p)

all_divs = [KL, reversed_KL, Jeffreys, JS]