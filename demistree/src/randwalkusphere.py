import numpy as np
from numba import jit

from . import linalg


@jit(nopython=True)
def usphere_rotate(phi, theta, phi_start, theta_start, phi_end, theta_end):
    # Convert spherical coordinates to Cartesian coordinates
    u = np.array([
        np.cos(phi_start) * np.sin(theta_start),
        np.sin(phi_start) * np.sin(theta_start),
        np.cos(theta_start)
    ])
    v = np.array([
        np.cos(phi_end) * np.sin(theta_end),
        np.sin(phi_end) * np.sin(theta_end),
        np.cos(theta_end)
    ])

    n1 = linalg.crossvector3(u, v)
    n = linalg.normalisevector(n1)

    t = linalg.crossvector3(n, u)

    dot_v_t = linalg.dotvector3(v, t)
    dot_v_u = linalg.dotvector3(v, u)

    alpha = np.arctan2(dot_v_t, dot_v_u)

    rmatrix = np.array([
        np.cos(alpha), -np.sin(alpha), 0,
        np.sin(alpha), np.cos(alpha), 0,
        0, 0, 1
    ]).reshape(3, 3)

    tmatrix = np.array([
        u[0], t[0], n[0],
        u[1], t[1], n[1],
        u[2], t[2], n[2]
    ]).reshape(3, 3)

    inv_tmatrix = linalg.inv3by3(tmatrix)

    inpos = np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta)
    ])

    outpos1 = linalg.dot3by3mat3vec(inv_tmatrix, inpos)
    outpos2 = linalg.dot3by3mat3vec(rmatrix, outpos1)
    outpos = linalg.dot3by3mat3vec(tmatrix, outpos2)

    r = np.sqrt(np.dot(outpos, outpos))
    phi_new = np.arctan2(outpos[1], outpos[0])
    if phi_new < 0:
        phi_new += 2 * np.pi
    theta_new = np.arccos(outpos[2] / r)

    return phi_new, theta_new


@jit(nopython=True)
def rand_walk_usphere(steps, prand, phi0, theta0):
    length = len(steps) + 1
    phi = np.zeros(length)
    theta = np.zeros(length)

    phi[0] = phi0
    theta[0] = theta0

    phinow = phi0
    thetanow = theta0

    for i in range(1, length):
        dphi = prand[i - 1]
        dtheta = steps[i - 1]

        phinew, thetanew = usphere_rotate(dphi, dtheta, 0, 0, phinow, thetanow)

        phinow = phinew
        thetanow = thetanew

        phi[i] = phinow
        theta[i] = thetanow

    return phi, theta
