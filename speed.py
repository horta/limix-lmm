from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import (
    newaxis,
    block,
    dot,
    eye,
    add,
    kron,
    concatenate,
    array,
    zeros,
    triu_indices_from,
    diag,
    full,
    asarray,
    sqrt,
)

from numpy_sugar.linalg import rsolve, economic_qs, dotd

from limix_lmm._blk_diag import BlockDiag

import numpy as np

np.set_printoptions(precision=20)


def fit_beta(Y, A, M, C0, C1, QS, G):
    Q = QS[0][0]
    S = QS[1]
    # Assume full-rank
    assert Q.shape[1] == S.shape[0]

    n, p = Y.shape
    d = M.shape[1]
    s = G.shape[1]

    hM = dot(Q.T, M)
    hG = dot(Q.T, G)
    QtY = _vec(dot(Q.T, Y))

    D12 = get_D12(C0, C1, QS).inv()
    assert D12.shape == (n * p, n * p)

    u = D12.dot(QtY)

    aM = D12.dot_kron(A, hM)
    aG = D12.dot_kron(A, hG)

    assert aM.shape == (n * p, p * d)
    assert aG.shape == (n * p, p * s)

    oM = dot(u.T, aM)
    oG = dot(u.T, aG)

    assert oM.shape == (1, p * d)
    assert oG.shape == (1, p * s)

    MM = dot(aM.T, aM)
    MG = dot(aM.T, aG)
    GG = compute_gg(aG, n, s, p)
    assert GG.shape == (p, p, p * s)

    assert MM.shape == (p * d, p * d)
    assert MG.shape == (p * d, p * s)
    # assert GG.shape == (p * n, p * n)

    # compute_lhs(MM, MG, GG, n, p, d, s)

    return []


def compute_lhs(MM, MG, GG, n, p, d, s):
    L = zeros((p * (d + 1), p * (d + 1)))
    for i0 in range(p):
        for i1 in range(p):
            u = _get_blk(L, i0, i1, d + 1)
            ul = u[:-1][:, :-1]
            ul[:] = _get_blk(MM, i0, i1, d)

            ur = u[:-1][:, [-1]]
            # ur[:] = _get_blk(MG, i0, i1, d, s)

            ll = u[[-1]][:, :-1]
            # ll[:] = ur[:].T

            lr = u[[-1]][:, [-1]]

            lr[:] = _get_blk(GG, i0, i1, 1, 1)


def compute_gg(aG, n, s, p):
    GG = zeros((p, p, p * s))
    for i in range(p):
        a = _row_blk(aG, i, n)
        for j in range(p):
            b = _row_blk(aG, j, n)
            GG[i, j, :] = dotd(a.T, b)
    return GG


def _get_blk(A, i, j, lblk_size, rblk_size=None):
    if rblk_size is None:
        rblk_size = lblk_size
    A = A[i * lblk_size : (i + 1) * lblk_size, :]
    A = A[:, j * rblk_size : (j + 1) * rblk_size]
    return A


def _row_blk(A, blk, blk_size):
    return A[blk * blk_size : (blk + 1) * blk_size]


def compute_chunks(G):
    size = 2
    chunks = [size] * (G.shape[1] // size)
    if G.shape[1] % size > 0:
        chunks.append(G.shape[1] % size)
    return chunks


def get_D12(C0, C1, QS):
    C0 = C0
    C1 = C1
    S0 = QS[1]

    p = C0.shape[0]
    n = QS[0][0].shape[0]
    r = QS[0][0].shape[1]

    siz = [s for s in [r, n - r] if s > 0]
    D = [BlockDiag(p, s) for s in siz]

    for i in range(C0.shape[0]):
        for j in range(C0.shape[1]):
            D[0].set_block(i, j, C0[i, j] * S0 + C1[i, j])
            if len(D) > 1:
                D[1].set_block(i, j, full(n - r, C1[i, j]))

    D = D[0]
    D._data = sqrt(D._data)

    return D


def _vec(X):
    X = asarray(X, float)
    return X.reshape((-1, 1), order="F")


def test():

    random = RandomState(0)
    # samples
    n = 5
    # traits
    p = 2
    # covariates
    d = 3

    Y = random.randn(n, p)
    A = random.randn(p, p)
    A = dot(A, A.T)
    M = random.randn(n, d)
    K = random.randn(n, n)
    K = (K - K.mean(0)) / K.std(0)
    K = K.dot(K.T) + eye(n) + 1e-3
    QS = economic_qs(K)

    C0 = random.randn(p, p)
    C0 = dot(C0, C0.T)
    C1 = random.randn(p, p)
    C1 = dot(C1, C1.T)
    G = random.randn(n, 3)

    betas = fit_beta(Y, A, M, C0, C1, QS, G)

    assert_allclose(
        betas,
        [
            array(
                [
                    [0.1171543072226424],
                    [0.2922669722595269],
                    [-0.02153087832329973],
                    [-0.6785191889622902],
                    [1.2163628766377277],
                    [-0.1328747439139128],
                    [-0.7187298358085206],
                    [-1.3501558521634132],
                ]
            ).reshape((-1, p), order="F"),
            array(
                [
                    [-0.38239934605314946],
                    [0.24597204056173463],
                    [0.010946258320120424],
                    [-0.04119008869431426],
                    [0.1474223136659856],
                    [-0.3345533712484771],
                    [-1.4415249194182163],
                    [-1.490028121254687],
                ]
            ).reshape((-1, p), order="F"),
            array(
                [
                    [0.22472471155023616],
                    [0.7345724052293824],
                    [0.18207580059536876],
                    [0.5916437252056872],
                    [1.2864372666081683],
                    [0.5670883175815873],
                    [-0.3512789451485551],
                    [0.9050459221116203],
                ]
            ).reshape((-1, p), order="F"),
        ],
    )


def single_snp():

    random = RandomState(0)
    # samples
    n = 5
    # traits
    p = 2
    # covariates
    d = 3
    # snps
    s = 1

    Y = random.randn(n, p)
    A = random.randn(p, p)
    A = dot(A, A.T)
    M = random.randn(n, d)
    K = random.randn(n, n)
    K = (K - K.mean(0)) / K.std(0)
    K = K.dot(K.T) + eye(n) + 1e-3
    QS = economic_qs(K)

    C0 = random.randn(p, p)
    C0 = dot(C0, C0.T)
    C1 = random.randn(p, p)
    C1 = dot(C1, C1.T)
    G = random.randn(n, s)

    betas = fit_beta(Y, A, M, C0, C1, QS, G)

    assert_allclose(
        betas,
        [
            [
                [-0.37773647655749726, -0.1461512668078721],
                [0.39619583249297224, -1.0620842567135134],
                [0.2696841276251685, -1.080005990730115],
                [0.14900547377519524, -1.5949226353531243],
            ]
        ],
    )


def slow():
    random = RandomState(0)
    # samples
    n = 1000
    # traits
    p = 2
    # covariates
    d = 1
    # SNPs
    s = 1000

    Y = random.randn(n, p)
    A = random.randn(p, p)
    A = dot(A, A.T)
    M = random.randn(n, d)
    K = random.randn(n, n)
    K = (K - K.mean(0)) / K.std(0)
    K = K.dot(K.T) + eye(n) + 1e-3
    QS = economic_qs(K)

    C0 = random.randn(p, p)
    C0 = dot(C0, C0.T)
    C1 = random.randn(p, p)
    C1 = dot(C1, C1.T)
    G = random.randn(n, s)

    betas = fit_beta(Y, A, M, C0, C1, QS, G)


def combine(A, B, p):
    n = A.shape[0]
    A = A.reshape((n, -1, p), order="F")
    B = B.reshape((n, -1, p), order="F")
    return concatenate([A, B], axis=1).reshape((n, -1), order="F")


if __name__ == "__main__":
    # single_snp()
    test()
    # slow()
    # 2.37 s ± 28.9 ms per loop
    # 1.57 s ± 32.2 ms per loop
