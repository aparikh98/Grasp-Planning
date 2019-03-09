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
import utils
#import tf.transformations as tfs

# 106B lab imports
from lab2.metrics import (
    compute_force_closure,
    compute_gravity_resistance,
    compute_custom_metric
)
from lab2.utils import *

# YOUR CODE HERE
# probably don't need to change these (BUT confirm that they're correct)
MAX_HAND_DISTANCE = .07
MIN_HAND_DISTANCE = .03
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1
TABLE_Z =  -0.201
COM_X =  0.688 
COM_Z =  -0.112 

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
        self.n_runs = 1
        # This is a function, one of the functions in src/lab2/metrics/metrics.py
        self.metric = eval(metric_name)

    def vertices_to_baxter_hand_pose(self, grasp_vertices, approach_directions):
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
        T_grasp_worlds = []
        #need to call
        Rs = []
        for grasps_vertex, approach_direction in zip(grasp_vertices, approach_directions):
            t = np.mean(grasps_vertex, axis = 0)
            line = (grasps_vertex[1] -grasps_vertex[0])
            R = utils.look_at_general_2(t, approach_direction, line)
            T_grasp_world = RigidTransform(rotation=R[:3, :3], translation = t)
            T_grasp_worlds.append(T_grasp_world)
            Rs.append(R[:3, :3])
        return T_grasp_worlds, Rs



    def sample_grasps(self, vertices, normals, mesh):
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
        # TODO when we find the high of the table vs real world position
        com = mesh.center_mass
        new_vertices = []
        new_normals = []

        for vertex, normal in zip(vertices,normals):
            angle = np.arctan((vertex[2]- com[2] )/(com[0] - vertex[0]))
            if vertex[2] > com[2]  and vertex[0] < com[0] and  angle >= np.pi/4 and angle <= np.pi/2 : # vertice is 3cm over table
                new_vertices.append(vertex)
                new_normals.append(normal)
        new_vertices = np.array(new_vertices)
        new_normals = np.array(new_normals)

        n = len(new_vertices)

        num_grasps = 100

        grasp_vertices = []
        grasp_normals = []

        i = 0
        iter = 0
        MAX_ITER = 1000
        while (i < num_grasps and iter < MAX_ITER):
            c1 = np.random.randint(0,n,1)
            c2 = np.random.randint(0,n,1)

            vertex1 = new_vertices[c1,:].flatten()
            vertex2 = new_vertices[c2,:].flatten()

            distance = np.linalg.norm(vertex1 - vertex2)

            if distance > MIN_HAND_DISTANCE and distance < MAX_HAND_DISTANCE:
                grasp_vertices.append([vertex1, vertex2])
                grasp_normals.append([new_normals[c1,:].flatten(),new_normals[c2,:].flatten()])
                i += 1
            iter += 1

        return np.asarray(grasp_vertices), np.asarray(grasp_normals)

    def score_grasps(self, grasp_vertices, grasp_normals, object_mass, mesh):
        """
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
            score = self.metric(grasp_vertice, grasp_normal, self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, mesh, self.n_runs)
            #score = np.random.rand() + 0.5
            scores.append(score)

        return np.asarray(scores)

    def vis(self, mesh, grasp_vertices, grasp_qualities,top_n_grasp_vertices, approach_directions, rs, TG):
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

        for i, (grasp, quality) in enumerate(zip(grasp_vertices, grasp_qualities)):
            color = [min(1, 2 * (1 - quality)), min(1, 2 * quality), 0, 1]
            vis3d.plot3d(grasp, color=color, tube_radius=.001)

        blue = [0, 0,255]
        light_blue = [50, 50,200]

        for i, (grasp, approach_direction) in enumerate(zip(top_n_grasp_vertices,approach_directions)):
            midpoint = np.mean(grasp,axis=0)
            approach_direction = np.asarray([midpoint, midpoint - 0.1 * approach_direction])
            if (i == 0):
                vis3d.plot3d(approach_direction, color=blue, tube_radius=.005)
                vis3d.points(grasp, color=blue, scale=.005)
            else:
                vis3d.plot3d(approach_direction, color=light_blue, tube_radius=.001)
                vis3d.points(grasp, color=light_blue, scale=.001)

        midpoint = np.mean(top_n_grasp_vertices[0], axis = 0)
        x_purp = [204, 0,204]
        x_axis = np.asarray([midpoint, midpoint + 0.1 *rs[:,0]])
        y_cyan = [0, 204,204]
        y_axis = np.asarray([midpoint, midpoint + 0.1 *rs[:, 1]])
        z_black = [0, 0,0]
        z_axis = np.asarray([midpoint, midpoint + 0.3 *rs[:, 2]])

        vis3d.plot3d(x_axis, color=x_purp, tube_radius=.005)
        vis3d.plot3d(y_axis, color=y_cyan, tube_radius=.005)
        vis3d.plot3d(z_axis, color=z_black, tube_radius=.005)

        origin = np.asarray([0,0,0])
        x_axis = np.asarray([origin, 0.2 * np.array([1, 0,0])])
        y_axis = np.asarray([origin, 0.2 * np.array([0, 1,0])])
        z_axis = np.asarray([origin, 0.2 * np.array([0, 0,1])])

        vis3d.plot3d(x_axis, color=x_purp, tube_radius=.005)
        vis3d.plot3d(y_axis, color=y_cyan, tube_radius=.005)
        vis3d.plot3d(z_axis, color=z_black, tube_radius=.005)

        # eucl_orien = np.asarray(tfs.euler_from_quaternion(TG.quaternion))
        intermediate_pos = TG.position - np.reshape(np.matmul(TG.rotation , np.array([[0], [0], [0.2]])), (1,3))
        print(intermediate_pos, np.shape(intermediate_pos))
        red = [255, 0,0]
        vis3d.points(intermediate_pos, color=red, scale=.01)
        vis3d.points(TG.position, color=red, scale=.01)



        vis3d.show()


    def calculate_normals(self,vertices, faces):
        # source for calculating normals
        # For each face we calculate a normal
        normals = []
        for triangle in faces:
            P0 = vertices[triangle[0]]
            P1 = vertices[triangle[1]]
            P2 = vertices[triangle[2]]

            normal = np.cross(P0 - P1, P0 - P2)
            normal = normal / np.linalg.norm(normal)

            normals.append(normal)

        normals = np.array(normals)

        return normals

    def calculate_pos_approach(self, grasp_vertices, com):
        directions = []
        positions = []
        for grasp_vertex in grasp_vertices:
            midpoint = np.mean(grasp_vertex, axis =0)
            direction = normalize(com - midpoint)
            directions.append(direction)
            positions.append(midpoint)
        return positions, directions


    def top_n_actions(self, mesh, obj_name, vis=False):
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

        samples, face_index = trimesh.sample.sample_surface_even(mesh, 500)
        com = mesh.center_mass
        print(com)
        vertices = mesh.vertices
        faces = mesh.faces
        face_normals = mesh.face_normals

        #normals = self.calculate_normals(vertices,faces)
        normals = face_normals[face_index]

        grasp_vertices, grasp_normals = self.sample_grasps(samples,normals, mesh)
        grasp_qualities = self.score_grasps(grasp_vertices,grasp_normals,OBJECT_MASS[obj_name], mesh)

        print (sum(grasp_qualities), len(grasp_qualities))

        top_n_idx = np.argsort(grasp_qualities)[-topN:][::-1]
        top_n_grasp_vertices = [grasp_vertices[i] for i in top_n_idx]
        top_n_grasp_normals = [grasp_normals[i] for i in top_n_idx]
        top_n_grasp_scores = [grasp_qualities[i] for i in top_n_idx] #maybe we will need this...
        print("The top n grasp scores are: ")
        for score in top_n_grasp_scores:
            print(score)
        print (sum(top_n_grasp_scores)/ len(top_n_grasp_scores))

        # How should we think about the approach direction
        # self.calculate_approach(top_n_grasp_vertices, com)
        positions, approach_directions = self.calculate_pos_approach(top_n_grasp_vertices, com)

        # np.mean(np.array(top_n_grasp_normals),axis=1)
        T_grasp_worlds, Rs =  self.vertices_to_baxter_hand_pose(top_n_grasp_vertices, approach_directions)
        if vis:
            self.vis(mesh, grasp_vertices, grasp_qualities, top_n_grasp_vertices, approach_directions, Rs[0], T_grasp_worlds[0])
        self.n_runs += 1
        return T_grasp_worlds
        # return zip(positions, approach_directions)
         
