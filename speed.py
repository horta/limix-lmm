from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy import dot, eye, add, kron, concatenate, array, zeros, triu_indices_from

from numpy_sugar.linalg import rsolve, economic_qs

from limix_lmm._blk_diag import BlockDiag

import numpy as np

np.set_printoptions(precision=20)


def fit_beta(Y, A, M, C0, C1, QS, G):
    n, p = Y.shape

    Di = [D.inv() for D in D(C0, C1, QS)]
    QtY = [dot(Q.T, Y) for Q in QS[0] if Q.size > 0]
    DiQtY = [Di.dot_vec(QtY) for Di, QtY in zip(Di, QtY)]

    QtM = [dot(Q.T, M) for Q in QS[0] if Q.size > 0]

    AQtM = [kron(A, QtM) for QtM in QtM]

    DiAQtM = [Di.dot(AQtM) for Di, AQtM in zip(Di, AQtM)]

    MQADQY = [
        dot(i.T, j).reshape((M.shape[1], -1), order="F") for i, j in zip(AQtM, DiQtY)
    ]

    chunks = compute_chunks(G)

    start = 0
    betas = []
    for chunk in chunks:
        end = start + chunk
        betas += fit_beta_chunk(G[:, start:end], A, QS, AQtM, Di, DiQtY, DiAQtM, MQADQY)
        start = end

    return betas


def fit_beta_chunk(G, A, QS, AQtM, Di, DiQtY, DiAQtM, MQADQY):

    betas = []
    p = A.shape[0]
    for j in range(G.shape[1]):

        rows = []
        AQtG = [kron(A, dot(Q.T, G[:, [j]])) for Q in QS[0]]
        AQtMG = [combine(i, j, p) for i, j in zip(AQtM, AQtG)]
        DiAQtG = [Di.dot(AQtG) for Di, AQtG in zip(Di, AQtG)]
        DiAQtMG = [combine(i, j, p) for i, j in zip(DiAQtM, DiAQtG)]

        L = DiAQtMG[0]
        R = AQtMG[0]
        siz = L.shape[1] // p
        d = AQtM[0].shape[1] // p

        for i in range(p):
            Li = L[:, i * siz : i * siz + siz]
            Ri = R[:, i * siz : i * siz + siz]
            Lim = Li[:, :d]
            Rim = Ri[:, :d]
            Lig = Li[:, d:]
            Rig = Ri[:, d:]
            LimRim = dot(Lim.T, Rim)
            LimRig = dot(Lim.T, Rig[:, [0]])
            LigRig = dot(Lig[:, [0]].T, Rig[:, [0]])

            # row0 = [LimRim, LimRig]
            # row1 = [LimRig.T, LigRig]
            row0 = []
            row1 = []
            for l in range(i, p):
                Ll = L[:, l * siz : l * siz + siz]
                Rl = R[:, l * siz : l * siz + siz]
                # Llm = Ll[:, :d]
                Rlm = Rl[:, :d]
                # Llg = Ll[:, d:]
                Rlg = Rl[:, d:]
                # LlmRlm = dot(Llm.T, Rlm)
                LimRlm = dot(Lim.T, Rlm)
                LimRlg = dot(Lim.T, Rlg[:, [0]])
                LigRlg = dot(Lig[:, [0]].T, Rlg[:, [0]])

                row0 += [LimRlm, LimRlg]
                row1 += [LimRlg.T, LigRlg]

            rows.append(row0)
            rows.append(row1)

            # LimRig = dot(Lim.T, Rig[:, [j]])
            # LigRig = dot(Lig[:, [j]].T, Rig[:, [j]])

            # LlmRlg = dot(Llm.T, Rlg[:, [j]])
            # LlgRlg = dot(Llg[:, [j]].T, Rlg[:, [j]])

            # denomi[(i, l, j)] = block(
            #     [
            #         [LimRim, LimRig, LimRlm, LimRlg],
            #         [LimRig.T, LigRig, LimRlg.T, LigRlg],
            #         [LimRlm.T, LimRlg, LlmRlm, LlmRlg],
            #         [LimRlg.T, LigRlg.T, LlmRlg.T, LlgRlg],
            #     ]
            # )

        ncols = sum([r.shape[1] for r in rows[0]])
        nrows = ncols
        deno = zeros((nrows, ncols))
        row_offset = 0
        col_offset_start = 0
        for ii in range(len(rows) // 2):
            row0 = rows[ii * 2]
            row1 = rows[ii * 2 + 1]
            col_offset = col_offset_start
            for jj in range(len(row1)):
                r0, c0 = row0[jj].shape
                deno[row_offset : row_offset + r0][:, col_offset : col_offset + c0] = row0[
                    jj
                ]

                r1, c1 = row1[jj].shape
                deno[row_offset + r0 : row_offset + r0 + r1][
                    :, col_offset : col_offset + c1
                ] = row1[jj]

                col_offset += c0

            col_offset_start += row0[0].shape[1] + row0[1].shape[1]
            row_offset += row0[0].shape[0] + row1[0].shape[0]

        inds = triu_indices_from(deno, k=1)
        deno[(inds[1], inds[0])] = deno[inds]

        # _denominator = outer(denomi[(0, 0)], denomi[(1, 0)])
        denominator = [deno]
        # denominator = [dot(i.T, j) for i, j in zip(DiAQtMG, AQtMG)]

        GQADQY = [
            dot(ii.T, j).reshape((1, -1), order="F") for ii, j in zip(AQtG, DiQtY)
        ]
        MQADQY = [i.reshape((d, -1), order="F") for i in MQADQY]
        nominator = [concatenate([i, j], axis=0) for i, j in zip(MQADQY, GQADQY)]
        nominator = [i.reshape((-1, 1), order="F") for i in nominator]

        denominator = add.reduce(denominator)
        nominator = add.reduce(nominator)

        beta = rsolve(denominator, nominator).reshape((-1, p), order="F")
        betas.append(beta)
    return betas


def compute_chunks(G):
    size = 2
    chunks = [size] * (G.shape[1] // size)
    if G.shape[1] % size > 0:
        chunks.append(G.shape[1] % size)
    return chunks


def D(C0, C1, QS):
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

    return D


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
    single_snp()
    test()
    # slow()
    # 2.37 s ± 28.9 ms per loop
    # 1.57 s ± 32.2 ms per loop
