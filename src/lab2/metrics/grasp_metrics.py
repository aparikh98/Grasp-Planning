#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasp Metrics for EE106B grasp planning lab
Author: Chris Correa
"""
# may need more imports
import numpy as np
from lab2.utils import vec, adj, look_at_general
# import cvxpy as cvx
import math
import osqp
import scipy.sparse as sparse
from numpy.linalg import norm
# from trimesh.ray import ray_triangle as rt

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
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

    # Find tangents and store them in a rotation matrix.
    R1 = look_at_general(vertices[0], normals[0])
    R2 = look_at_general(vertices[1], normals[1])

    theta = 0.2 # We should be able to choose this independently, but it changes the results..
    f1 = normals[0] + np.cos(theta)* R1[0:3,0] +np.sin(theta)* R1[0:3,1]
    f2 = normals[1] + np.cos(theta)* R2[0:3,0] +np.sin(theta)* R2[0:3,1]

    line = vertices[1,:] - vertices[0,:]

    # we compare the angles between the line and the normal with the angle between fc vector and the normal
    line_in_FC1 = abs(np.matmul(line, normals[0]) / (norm(line) * norm(normals[0]))) > \
                  abs(np.matmul(line, f1) / (norm(line) * norm(f1)))

    line_in_FC2 = abs(np.matmul(-line, normals[1]) / (norm(-line) * norm(normals[1]))) > \
                  abs(np.matmul(-line, f2) / (norm(-line) * norm(f2)))

    return line_in_FC1 and line_in_FC2

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

    return G, np.matmul(A1,B), np.matmul(A2,B)

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

    G, G1, G2 = get_grasp_map(vertices, normals, num_facets, mu, gamma)

    P11 = np.matmul(G1.T, G1)
    P12 = np.matmul(G1.T, G2)
    P21 = np.matmul(G2.T, G1)
    P22 = np.matmul(G2.T, G2)

    C = np.zeros( (4,4) )

    P = sparse.csc_matrix(np.vstack((np.hstack((P11,C,P12,C)),
                  np.hstack((C,C,C,C)),
                  np.hstack((P21,C,P22,C)),
                  np.hstack((P11,C,P12,C)))))

    q = - np.vstack((np.matmul(G1.T,desired_wrench),
                    np.zeros((4,1)),
                    np.matmul(G2.T, desired_wrench),
                    np.zeros((4,1))))

    #P = sparse.csc_matrix(np.matmul(np.transpose(G),G))
    #q = np.asarray(np.matmul(-G.T, desired_wrench))

    A = np.matrix([[0,0,-mu,0, 1, 1,0,0],
                         [1,0,0,0,-1,0,0,0],
                         [0,1,0,0,0,-1,0,0],
                         [-1,0,0,0,-1,0,0,0],
                         [0,-1,0,0,-1,0,0,0],
                         [0,0,-1,0,0,0,0,0],
                         [0,0,0,1,0,0,-1,0],
                         [0, 0, 0, -1, 0, 0, -1,0],
                         [0, 0, -gamma, 0, 0, 0, 1,0]])


    A =sparse.csc_matrix(np.vstack((np.hstack((A, np.zeros_like(A))),
                  np.hstack((np.zeros_like(A), A)))))

    l = np.asarray(-np.inf*np.ones((18,1)))
    u = np.asarray(np.zeros((18,1)))

    m = osqp.OSQP()

    m.setup(P=P, q=q, A=A, l=l, u= u)
    results = m.solve()

    epsilon = 0.01

    f = np.asarray([results.x[0],results.x[1],results.x[2], results.x[3],
                    results.x[8],results.x[9],results.x[10], results.x[11]]).reshape((8,1))

    if np.linalg.norm(np.matmul(G,f) - desired_wrench) <= epsilon:
        return True # There exists a correct solution
    else:
        return False

def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
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

    g = np.array([0,0,-9.81 * object_mass,0,0,0]).reshape((6,1))

    return contact_forces_exist(vertices, normals, num_facets, mu, gamma, g)

def compute_custom_metric(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
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

    scale_mu = 0.4
    scale_gamma = 0.5
    scale_vertices = 0.01
    scale_normals = 0.01

    num_experiments = 100
    avg_force_close = 0.0
    num_count = 0

    # Ray_object = rt.RayMeshIntersector(mesh)

    for i in range(num_experiments):
        mu_noise = np.random.normal(loc = mu, scale = scale_mu)
        gamma_noise = np.random.normal(loc = gamma, scale = scale_gamma)
        vertices_noise = np.random.normal(loc = vertices, scale = scale_vertices)

        print(vertices_noise[0,:], np.shape(vertices_noise[0,:]), mesh.center_mass, np.shape(mesh.center_mass))

        vertex_0, index_ray0, index_tri_0 = mesh.ray.intersects_location(np.asarray(vertices_noise[0,:]), np.asarray(mesh.center_mass) - np.asarray(vertices_noise[0,:]))
        if len(vertex_0) == 0 or np.linalg.norm(vertex_0[0], vertices_noise[0]) >= scale_vertices * 3: 
            continue

        normal_0 = mesh.face_normals[index_tri_0[0]]
   
        print(vertex_0, np.shape(vertex_0), normal_0, np.shape(normal_0), mesh.center_mass, np.shape(mesh.center_mass))

        vertex_1, index_ray1, index_tri_1 = mesh.ray.intersects_location(np.asarray(vertices_noise[1,:]), np.asarray(mesh.center_mass) - np.asarray(vertices_noise[1,:]))
        if len(vertex_1) == 0 or np.linalg.norm(vertex_1[0], vertices_noise[1]) >= scale_vertices * 3: 
            continue

        normal_1 = mesh.face_normals[index_tri_1[0]]
        new_vertices = np.asarray([vertex_0[0], vertex_1[0]])

        force_closure = compute_force_closure(new_vertices, new_normals, num_facets, mu_noise, gamma_noise, object_mass)
        num_count += 1
        if force_closure:
            avg_force_close += 1.0
    print(num_count)
    return (avg_force_close / num_count)
