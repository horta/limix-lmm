from numpy import dot, sqrt, eye, tril, abs, diag, kron, zeros_like, block
from numpy.testing import assert_allclose
from numpy.random import RandomState
from numpy import set_printoptions


def triang(D, bsiz):
    r""" Block triangularisation.

    I want to compute

    L = | L₀   0 |
        | L₁  L₂ |

    such that LLᵗ = D, for which L and D are
    square, recursively-diagonal block matrices.

    Let

    | L₀   0 |   | L₀  L₁ᵗ |     | D₀  D₁ |
    | L₁  L₂ | × |  0  L₂  |  =  | D₁  D₂ |

    where L₀ and L₂ are square matrices. We have

    L₀ = √(D₀)
    L₁ = [ L₀⁻¹D₀₀  L₀⁻¹D₀₁ ]
    L₂ = √(D₂ - L₁ᵗL₁)

    for which √(X) = Y such that YYᵗ = X. By partitioning
    D such that D₀ is a diagonal, square matrix we trivially
    have L₀ and L₁. L₂ is found by recursively applying the
    same algorithm.
    """
    if D.shape[0] == bsiz:
        return sqrt(D)

    D0 = D[:bsiz][:, :bsiz]
    D1 = D[:bsiz][:, bsiz:]
    D2 = D[bsiz:][:, bsiz:]

    L0 = sqrt(D0)
    L1 = (D1.T / L0.diagonal()).T
    L2 = triang(D2 - dot(L1.T, L1), bsiz)

    zero = zeros_like(L1)
    return block([[L0, zero], [L1.T, L2]])


if __name__ == "__main__":
    set_printoptions(precision=2)
    for i in range(10):
        random = RandomState(i)

        n = 5
        p = 2

        C0 = tril(random.randn(p, p))
        C0 = C0.dot(C0.T)

        C1 = tril(random.randn(p, p))
        C1 = C1.dot(C1.T)

        S = diag(abs(random.randn(n)))

        D = kron(C0, S) + kron(C1, eye(n))
        L = triang(D, n)
        assert_allclose(abs(dot(L, L.T) - D).min(), 0, atol=1e-9)

        random = RandomState(i)

        n = 5
        p = 3

        C0 = tril(random.randn(p, p))
        C0 = C0.dot(C0.T)

        C1 = tril(random.randn(p, p))
        C1 = C1.dot(C1.T)

        S = diag(abs(random.randn(n)))

        D = kron(C0, S) + kron(C1, eye(n))
        L = triang(D, n)
        assert_allclose(abs(dot(L, L.T) - D).min(), 0, atol=1e-9)
