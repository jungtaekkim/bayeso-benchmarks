#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: October 27, 2021
#

import numpy as np


def pdf_two_dim_normal(bx, mu, Cov):
    assert bx.shape[0] == mu.shape[0] == Cov.shape[0] == Cov.shape[1] == 2
    
    dim = bx.shape[0]

    term_first = ((2.0 * np.pi)**(-0.5 * dim)) * (np.linalg.det(Cov)**(-0.5))
    term_second = np.exp(-0.5 * np.dot(np.dot((bx - mu), np.linalg.inv(Cov)), bx - mu))
    
    return term_first * term_second
