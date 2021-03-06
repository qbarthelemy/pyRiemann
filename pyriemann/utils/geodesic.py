import numpy as np

from .base import sqrtm, invsqrtm, powm, logm, expm

###############################################################
# geodesic
###############################################################


def geodesic(A, B, alpha, metric='riemann'):
    """Return the matrix at the position alpha on the geodesic between A and B
    according to the metric:

    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid'
    :returns: the covariance matrix on the geodesic

    """  # noqa
    options = {'riemann': geodesic_riemann,
               'logeuclid': geodesic_logeuclid,
               'euclid': geodesic_euclid}
    C = options[metric](A, B, alpha)
    return C


def geodesic_riemann(A, B, alpha=0.5):
    r"""Return the matrix at the position alpha on the Riemannian geodesic
    between A and B:

    .. math::
        \mathbf{C} = \mathbf{A}^{1/2} \left( \mathbf{A}^{-1/2} \mathbf{B} \mathbf{A}^{-1/2} \right)^\alpha \mathbf{A}^{1/2}

    C is equal to A if alpha = 0 and B if alpha = 1

    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix

    """  # noqa
    sA = sqrtm(A)
    isA = invsqrtm(A)
    C = np.dot(np.dot(isA, B), isA)
    D = powm(C, alpha)
    E = np.dot(np.dot(sA, D), sA)
    return E


def geodesic_euclid(A, B, alpha=0.5):
    r"""Return the matrix at the position alpha on the Euclidean geodesic
    between A and B:

    .. math::
        \mathbf{C} = (1-\alpha) \mathbf{A} + \alpha \mathbf{B}

    C is equal to A if alpha = 0 and B if alpha = 1

    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix

    """
    return (1 - alpha) * A + alpha * B


def geodesic_logeuclid(A, B, alpha=0.5):
    r"""Return the matrix at the position alpha on the log-Euclidean geodesic
    between A and B  :

    .. math::
        \mathbf{C} =  \exp \left( (1-\alpha) \log(\mathbf{A}) + \alpha \log(\mathbf{B}) \right)

    C is equal to A if alpha = 0 and B if alpha = 1

    :param A: the first coavriance matrix
    :param B: the second coavriance matrix
    :param alpha: the position on the geodesic
    :returns: the covariance matrix

    """  # noqa
    return expm((1 - alpha) * logm(A) + alpha * logm(B))
