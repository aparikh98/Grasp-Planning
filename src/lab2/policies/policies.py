#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasping Policy for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np

# Autolab imports
from autolab_core import RigidTransform
import trimesh
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from visualization import Visualizer3D as vis3d

# 106B lab imports
from lab2.metrics import (
    compute_force_closure,
    compute_gravity_resistance,
    compute_custom_metric
)
from lab2.utils import length, normalize

# YOUR CODE HERE
# probably don't need to change these (BUT confirm that they're correct)
MAX_HAND_DISTANCE = .04
MIN_HAND_DISTANCE = .01
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1

# TODO
OBJECT_MASS = {'gearbox': .25, 'nozzle': .25, 'pawn': .25}


class GraspingPolicy():
    def __init__(self, n_vert, n_grasps, n_execute, n_facets, metric_name):
        """
        Parameters
        ----------
        n_vert : int
            We are sampling vertices on the surface of the object, and will use pairs of
            these vertices as grasp candidates
        n_grasps : int
            how many grasps to sample.  Each grasp is a pair of vertices
        n_execute : int
            how many grasps to return in policy.action()
        n_facets : int
            how many facets should be used to approximate the friction cone between the
            finger and the object
        metric_name : string
            name of one of the function in src/lab2/metrics/metrics.py
        """
        self.n_vert = n_vert
        self.n_grasps = n_grasps
        self.n_facets = n_facets
        # This is a function, one of the functions in src/lab2/metrics/metrics.py
        self.metric = eval(metric_name)

    def vertices_to_baxter_hand_pose(grasp_vertices, approach_direction):
        """
        takes the contacts positions in the object frame and returns the hand pose T_obj_gripper
        BE CAREFUL ABOUT THE FROM FRAME AND TO FRAME.  the RigidTransform class' frames are
        weird.

        Parameters
        ----------
        grasp_vertices : 2x3 :obj:`numpy.ndarray`
            position of the fingers in object frame
        approach_direction : 3x' :obj:`numpy.ndarray`
            there are multiple grasps that go through contact1 and contact2.  This describes which
            orientation the hand should be in

        Returns
        -------
        :obj:`autolab_core:RigidTransform` Hand pose in the object frame
        """

        #need to call autolab_core.RigidTransform(rotation=array([[ 1., 0., 0.], [ 0., 1., 0.], [ 0., 0., 1.]]),
        # translation=array([ 0., 0., 0.]), from_frame='unassigned', to_frame='world'))


        # YOUR CODE HERE
        raise NotImplementedError

    def sample_grasps(self, vertices, normals):
        """

        Samples a bunch of candidate grasps.  You should randomly choose pairs of vertices and throw out
        pairs which are too big for the gripper, or too close too the table.  You should throw out vertices
        which are lower than ~3cm of the table.  You may want to change this.  Returns the pairs of
        grasp vertices and grasp normals (the normals at the grasp vertices)

        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
            mesh vertices
        normals : nx3 :obj:`numpy.ndarray`
            mesh normals
        T_ar_object : :obj:`autolab_core.RigidTransform`
            transform from the AR tag on the paper to the object

        Returns
        -------
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps vertices.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector and there are n_grasps of them, hence the shape n_graspsx2x3
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        """

        """
        first throw out any vertices within 3cm of the table
        for range (x):
            pick two random vertices, check if they are within grasping size,
            if they are add them and normals to the arrays
            otherwise skip
        """
        new_vertices = []
        new_normals = []
        for vertice, normal in zip(vertices,normals):
            if vertice[2] > 0.03: # vertice is 3cm over table
                new_vertices.append(vertice)
                new_normals.append(normal)

        n = len(new_vertices)
        num_grasps = n

        grasp_vertices = []
        grasp_normals = []

        for i in range(num_grasps):
            c1 = np.random.uniform(0,n,1)
            c2 = np.random.uniform(0,n,1)

            vertice1 = new_vertices[c1,:]
            vertice2 = new_vertices[c2,:]

            distance = np.linalg.norm(vertice1,vertice2)
            if distance > MIN_HAND_DISTANCE and distance < MAX_HAND_DISTANCE:
                grasp_vertices.append([vertice1, vertice2])
                grasp_normals.append([normals[c1,:],normals[c2,:]])

        return grasp_vertices, grasp_normals

    def score_grasps(self, grasp_vertices, grasp_normals, object_mass):
        """
        this is pretty easy right, just call the metrics for each of the samples?

        takes mesh and returns pairs of contacts and the quality of grasp between the contacts, sorted by quality

        Parameters
        ----------
        grasp_vertices : n_graspsx2x3 :obj:`numpy.ndarray`
            grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        grasp_normals : mx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3

        Returns
        -------
        :obj:`list` of int
            grasp quality for each
        """
        """
        for each grasp:
            call:  metric(vertices, normals, num_facets, mu, gamma, object_mass)
            return list of these metrics
        """

        scores = []
        for grasp_vertice,grasp_normal in zip(grasp_vertices,grasp_normals):
            score = self.metric(grasp_vertice, grasp_normal, self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass)
            scores.append(score)

        return scores

    def vis(self, mesh, grasp_vertices, grasp_qualities):
        """
        Pass in any grasp and its associated grasp quality.  this function will plot
        each grasp on the object and plot the grasps as a bar between the points, with
        colored dots on the line endpoints representing the grasp quality associated
        with each grasp

        Parameters
        ----------
        mesh : :obj:`Trimesh`
        grasp_vertices : mx2x3 :obj:`numpy.ndarray`
            m grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, hence the shape mx2x3
        grasp_qualities : mx' :obj:`numpy.ndarray`
            vector of grasp qualities for each grasp
        """
        vis3d.mesh(mesh)

        dirs = normalize(grasp_vertices[:, 0] - grasp_vertices[:, 1], axis=1)
        midpoints = (grasp_vertices[:, 0] + grasp_vertices[:, 1]) / 2
        grasp_endpoints = np.zeros(grasp_vertices.shape)
        grasp_vertices[:, 0] = midpoints + dirs * MAX_HAND_DISTANCE / 2
        grasp_vertices[:, 1] = midpoints - dirs * MAX_HAND_DISTANCE / 2

        for grasp, quality in zip(grasp_vertices, grasp_qualities):
            color = [min(1, 2 * (1 - quality)), min(1, 2 * quality), 0, 1]
            vis3d.plot3d(grasp, color=color, tube_radius=.001)
        vis3d.show()


    def calculate_normals(self,vertices, faces):
        # source for calculating normals
        # https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy

        def normalize_v3(arr):
            ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
            lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)

            arr[:, 0] /= lens
            arr[:, 1] /= lens
            arr[:, 2] /= lens

            return arr

        # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        norm = np.zeros(faces.shape, dtype=vertices.dtype)
        # Create an indexed view into the vertex array using the array of three indices for triangles
        tris = vertices[faces]
        # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
        # we need to normalize these, so that our next step weights each normal equally.
        n = normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        norm[faces[:, 0]] += n
        norm[faces[:, 1]] += n
        norm[faces[:, 2]] += n
        norm = normalize_v3(norm)

        return norm

    def top_n_actions(self, mesh, obj_name, vis=True):
        """
        call score grasps and return top n

        Takes in a mesh, samples a bunch of grasps on the mesh, evaluates them using the
        metric given in the constructor, and returns the best grasps for the mesh.  SHOULD
        RETURN GRASPS IN ORDER OF THEIR GRASP QUALITY.

        You should try to use mesh.mass to get the mass of the object.  You should check the
        output of this, because from this
        https://github.com/BerkeleyAutomation/trimesh/blob/master/trimesh/base.py#L2203
        it would appear that the mass is approximated using the volume of the object.  If it
        is not returning reasonable results, you can manually weight the objects and store
        them in the dictionary at the top of the file.

        Parameters
        ----------
        mesh : :obj:`Trimesh`
        vis : bool
            Whether or not to visualize the top grasps

        Returns
        -------
        :obj:`list` of :obj:`autolab_core.RigidTransform`
            the matrices T_grasp_world, which represents the hand poses of the baxter / sawyer
            which would result in the fingers being placed at the vertices of the best grasps
        """
        # Some objects have vertices in odd places, so you should sample evenly across
        # the mesh to get nicer candidate grasp points using trimesh.sample.sample_surface_even()

        """
        Use: trimesh.sample.sample_surface_even(mesh, count) and then throw out disqualified pairs
            Parameters
            ---------
            mesh: Trimesh object
            count: number of points to return
            Returns
            ---------
            samples: (count,3) points in space on the surface of mesh
            face_index: (count,) indices of faces for each sampled point

            call sample_grasps(samples,normals ) we need to compute the normals
                returns grasps vertices and grasps normals both nx2x3
             score_grasps(sample_grasps)
             visualize if needed
             call vertices_to_baxter_hand_pose(grasps, approach_direction)

        """
        topN = 10

        samples, face_index = trimesh.sample.sample_surface_even(mesh, self.n_facets)

        vertices = mesh.vertices
        faces = mesh.faces

        normals = self.calculate_normals(vertices,faces)
        print(np.shape(faces), np.shape(normals), np.shape(vertices))
        faces = faces[face_index]
        normals = normals[faces]
        vertices = vertices[faces]

        print(np.shape(faces),np.shape(normals),np.shape(vertices))

        grasp_vertices, grasp_normals = self.sample_grasps(vertices,normals)
        grasp_qualities = self.score_grasps(grasp_vertices,grasp_normals,obj_name)

        if vis:
            self.vis(mesh, grasp_vertices, grasp_qualities)

        top_n_idx = np.argsort(grasp_qualities)[-topN:]
        top_n_grasp_vertices = [grasp_qualities[i] for i in top_n_idx]
        top_n_grasp_normals = [grasp_qualities[i] for i in top_n_idx]
        #top_n_grasp_scores = [grasp_qualities[i] for i in top_n_idx] #maybe we will need this...

        #self.vertices_to_baxter_hand_pose()

        return top_n_grasp_vertices, top_n_grasp_normals