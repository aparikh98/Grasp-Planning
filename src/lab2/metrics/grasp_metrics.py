#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasp Metrics for EE106B grasp planning lab
Author: Chris Correa
"""
# may need more imports
import numpy as np
from lab2.utils import vec, adj, look_at_general
import cvxpy as cvx
import math
import osqp
import scipy.sparse as sparse

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Compute the force closure of some object at contacts, with normal vectors
    stored in normals You can use the line method described in HW2.  if you do you
    will not need num_facets

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors
        will be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # YOUR CODE HERE
    """
    First get the line between two grasp points
    calculate the friction cones
    check if the line passes through both
    """

    m = num_facets
    f1 = np.zeros((m,3))
    f2 = np.zeros((m,3))

    # Find tangents and store them in a rotation matrix.
    R1 = look_at_general(vertices[0], normals[0])
    R2 = look_at_general(vertices[1], normals[1])

    for i in range(m):
        f1[i,:] = normals[0] + np.cos(2*np.pi*i/m)* R1[0:3,0] +np.sin(2*np.pi*i/m)* R1[0:3,1]
        f2[i,:] = normals[1] + np.cos(2*np.pi*i/m)* R2[0:3,0] +np.sin(2*np.pi*i/m)* R2[0:3,1]

    vec_c1_to_c2 = vertices[0,:] - vertices[1,:]
    vec_c2_to_c1 = vertices[1,:] - vertices[0,:]

    #TODO: try n times. lin.alg.solve. if positive combination exists, we have force closure.
    #TODO: quadprop solver: https://osqp.org/docs/interfaces/python.html#python-interface


    raise NotImplementedError


def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    """
    defined in the book on page 219.  Compute the grasp map given the contact
    points and their surface normals

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors
        will be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient

    Returns
    -------
    :obj:`numpy.ndarray` grasp map
    """
    # YOUR CODE HERE

    if mu == 0:
        print("Use point contact without friction")
    if gamma == 0:
        print("Do not use soft-finger bases")

    B = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,1]])

    g1 = look_at_general(vertices[0],normals[0])
    A1 = np.linalg.inv(adj(g1))

    g2 = look_at_general(vertices[1],normals[1])
    A2 = np.linalg.inv(adj(g2))

    # as everything is in world coordinates we don't need to transform coordinate systems with the adjoint.
    G = np.hstack((np.matmul(A1,B), np.matmul(A2,B)))

    return G


def contact_forces_exist(vertices, normals, num_facets, mu, gamma, desired_wrench):
    """
    Compute whether the given grasp (at contacts with surface normals) can produce
    the desired_wrench.  will be used for gravity resistance.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors
        will be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    desired_wrench : :obj:`numpy.ndarray`
        potential wrench to be produced

    Returns
    -------
    bool : whether contact forces can produce the desired_wrench on the object
    """

    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)


    P = sparse.csc_matrix(np.matmul(np.transpose(G),G))
    q = sparse.csc_matrix(np.matmul(-G.T, desired_wrench))
    A = sparse.csc_matrix([[0,0,-mu,0, 1, 1,0,0],
                             [1,0,0,0,-1,0,0,0],
                             [0,1,0,0,0,-1,0,0],
                             [-1,0,0,0,-1,0,0,0],
                             [0,-1,0,0,-1,0,0,0],
                             [0,0,-1,0,0,0,0,0],
                              [0,0,0,1,0,0,-1,0],
                           [0, 0, 0, -1, 0, 0, -1,0],
                           [0, 0, -gamma, 0, 0, 0, 1,0]])

    l = sparse.csc_matrix(-np.inf*np.ones((9,1)))
    u = sparse.csc_matrix(np.zeros((9,1)))

    m = osqp.OSQP()
    print(np.shape(P), np.shape(q), np.shape(A), np.shape(l), np.shape(u))
    m.setup(P=P, q=q.T, A=A, l=l, u= u)
    results = m.solve()

    epsilon = 0.01

    if np.linalg.norm(np.matmul(G,results.x[:4]) - desired_wrench) <= epsilon:
        return True # There exists a correct solution
    else:
        return False

def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Gravity produces some wrench on your object.  Computes whether the grasp can
    produce and equal and opposite wrench

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors will
        be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # Design the gravity wrench, call contact_forces_exist on that
    # YOUR CODE HERE (contact forces exist may be useful here)
    print()
    g = np.array([0,0,-9.81 * object_mass,0,0,0]).reshape((6,1))

    return contact_forces_exist(vertices, normals, num_facets, mu, gamma, g)

def compute_custom_metric(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    I suggest Ferrari Canny, but feel free to do anything other metric you find.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors will
        be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # YOUR CODE HERE :)
    raise NotImplementedError
