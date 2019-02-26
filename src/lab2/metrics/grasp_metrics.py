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
    f1 = np.zeros(m)
    f2 = np.zeros(m)

    # Find tangents and store them in a rotation matrix.
    R1 = look_at_general(vertices[0], normals[0])
    R2 = look_at_general(vertices[1], normals[1])

    for i in range(m):
        f1[i] = normals[0] + np.cos(2*np.pi*i/m)* R1[0:3,0] +np.sin(2*np.pi*i/m)* R1[0:3,1]
        f2[i] = normals[1] + np.cos(2*np.pi*i/m)* R2[0:3,0] +np.sin(2*np.pi*i/m)* R2[0:3,1]

    vec_c1_to_c2 = vertices[0,:] - vertices[1,:]
    vec_c2_to_c1 = vertices[1,:] - vertices[0,:]

    #TODO: try n times. lin.alg.solve. if positive combination exists, we have force closure.
    #TODO: quadprop solver: https://osqp.org/docs/interfaces/python.html#python-interface

    P =
    q =
    A =

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=0, u= np.inf)
    results = m.solve()

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

    B = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1],
                  [0,0,0],
                  [0,0,0],
                  [0,0,0]])

    # as everything is in world coordinates we don't need to transform coordinate systems with the adjoint.
    G = np.array([B, B])

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


    m = num_facets
    f1 = np.zeros(m)
    f2 = np.zeros(m)

    e1 = normals[0]
    e2 = normals[1]

    e1_z = [0,0,e1[2]]
    e2_z = [0,0,e2[2]]

    e1_x = [e1[0],0,0]
    e2_x = [e2[0],0,0]

    e1_y = [0,e1[1],0]
    e2_y = [0,e2[1],0]

    for i in range(m):
        f1[i] = e1_z + np.cos(2*np.pi*i/m)* e1_x +np.sin(2*np.pi*i/m)*e1_y
        f2[i] = e2_z + np.cos(2*np.pi*i/m)* e2_x +np.sin(2*np.pi*i/m)*e2_y

    f = np.concatenate((f1,f2),axis=None)
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)
    F = np.matmul(G,f)

    x = np.linalg.solve(F,desired_wrench)
    if np.allclose(np.dot(F,x),desired_wrench):
        return True
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

    g = np.array([0,0,-9.81 * object_mass,0,0,0])

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
