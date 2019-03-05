#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Starter script for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np
import scipy
import sys
import argparse

# 106B lab imports
import os
sys.path.append(os.getcwd().replace('scripts','') + '/src/')
sys.path.append(os.getcwd().replace('scripts','') + '/src/lab2')
print("sys path", sys.path)
from lab2.policies import GraspingPolicy

# AutoLab imports
from autolab_core import RigidTransform
import trimesh



try:
    import rospy
    import tf
    from baxter_interface import gripper as baxter_gripper
    from path_planner import PathPlanner
    import tf.transformations as tfs
    from geometry_msgs.msg import Pose, PoseStamped
    from moveit_msgs.msg import OrientationConstraint
    ros_enabled = True
except:
    print( 'Couldn\'t import ROS.  I assume you\'re running this on your laptop')
    ros_enabled = False


def lookup_transform(to_frame, from_frame='base'):
    """
    Returns the AR tag position in world coordinates

    Parameters
    ----------
    to_frame : string
        examples are: ar_marker_7, gearbox, pawn, ar_marker_3, etc
    from_frame : string
        lets be real, you're probably only going to use 'base'

    Returns
    -------
    :obj:`autolab_core.RigidTransform` AR tag position or object in world coordinates
    """
    # if not ros_enabled:
    print( 'I am the lookup transform function!  ' \
        + 'You\'re not using ROS, so I\'m returning the Identity Matrix.')
    return RigidTransform(to_frame=from_frame, from_frame=to_frame)
    # listener = tf.TransformListener()
    # attempts, max_attempts, rate = 0, 10, rospy.Rate(1.0)
    # while attempts < max_attempts:
    #     try:
    #         t = listener.getLatestCommonTime(from_frame, to_frame)
    #         tag_pos, tag_rot = listener.lookupTransform(
    #             from_frame, to_frame, t)
    #     except:
    #         rate.sleep()
    #         attempts += 1
    # rot = RigidTransform.rotation_from_quaternion(tag_rot)
    # return RigidTransform(rot, tag_pos, to_frame=from_frame, from_frame=to_frame)


def execute_grasp(T_grasp_world, planner, gripper):
    """
    I think we need to add some heuristics on how to approach the object.
    For example we should take gripper orientation and move back in that direction

    takes in the desired hand position relative to the object, finds the desired
    hand position in world coordinates.  Then moves the gripper from its starting
    orientation to some distance BEHIND the object, then move to the  hand pose
    in world coordinates, closes the gripper, then moves up.

    Parameters
    ----------
    T_grasp_world : :obj:`autolab_core.RigidTransform`
        desired position of gripper relative to the world frame
    """
    def close_gripper():
        """closes the gripper"""
        gripper.close(block=True)
        rospy.sleep(1.0)

    def open_gripper():
        """opens the gripper"""
        gripper.open(block=True)
        rospy.sleep(1.0)

    final_position = T_grasp_world.position
    final_quaternion = T_grasp_world.quaternion
    eucl_orientation = tfs.euler_from_quaternion(T_grasp_world.quaternion)
    
    final_pose = Pose()
    final_pose.position.x = final_position[0]
    final_pose.position.y = final_position[1]
    final_pose.position.z = final_position[2]
    final_pose.orientation.x = final_quaternion[0]
    final_pose.orientation.y = final_quaternion[1]
    final_pose.orientation.z = final_quaternion[2]
    final_pose.orientation.w = final_quaternion[3]
    print("Final Pose", final_pose)

    orien_const = OrientationConstraint()
    orien_const.link_name = "left_gripper";
    orien_const.header.frame_id = "base";
    orien_const.orientation = final_pose.orientation
    orien_const.absolute_x_axis_tolerance = 0.1
    orien_const.absolute_y_axis_tolerance = 0.1
    orien_const.absolute_z_axis_tolerance = 0.1
    orien_const.weight = 1.0
   
    alpha = 0.1
    intermediate_pose = Pose()
    intermediate_pose.position.x = final_position[0] - eucl_orientation[0] * alpha
    intermediate_pose.position.y = final_position[1]  - eucl_orientation[1] * alpha
    intermediate_pose.position.z = final_position[2] - eucl_orientation[2] * alpha
    intermediate_pose.orientation.x = final_quaternion[0]
    intermediate_pose.orientation.y = final_quaternion[1]
    intermediate_pose.orientation.z = final_quaternion[2]
    intermediate_pose.orientation.w = final_quaternion[3]

    print("intermediate Pose", intermediate_pose)

    return

    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')

    plan = planner.plan_to_pose(intermediate_pose, list())
    planner.execute_plan(intermediate_pose)

    inp = raw_input('Press <Enter> to open gripper, or \'exit\' to exit')
    open_gripper()

    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    plan2 = planner.plan_to_pose(final_pose, [orien_const])
    planner.execute_plan(plan2)

    inp = raw_input('Press <Enter> to close gripper, or \'exit\' to exit')
    close_gripper()

    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    plan3 = planner.plan_to_pose(intermediate_pose,[orien_const])
    planner.execute_plan(plan3)

    inp = raw_input('Press <Enter> to open gripper, or \'exit\' to exit')
    open_gripper()

    if inp == "exit":
        return


def parse_args():
    """
    Pretty self explanatory tbh
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', type=str, default='pawn', help="""Which Object you\'re trying to pick up.  Options: gearbox, nozzle, pawn.
        Default: gearbox"""
                        )
    parser.add_argument('-n_vert', type=int, default=1000, help='How many vertices you want to sample on the object surface.  Default: 1000'
                        )
    parser.add_argument('-n_facets', type=int, default=32, help="""You will approximate the friction cone as a set of n_facets vectors along
        the surface.  This way, to check if a vector is within the friction cone, all
        you have to do is check if that vector can be represented by a POSITIVE
        linear combination of the n_facets vectors.  Default: 32"""
                        )
    parser.add_argument('-n_grasps', type=int, default=500,
                        help='How many grasps you want to sample.  Default: 500')
    parser.add_argument('-n_execute', type=int, default=5,
                        help='How many grasps you want to execute.  Default: 5')
    parser.add_argument('-metric', '-m', type=str, default='compute_custom_metric', help="""Which grasp metric in grasp_metrics.py to use.
        Options: compute_force_closure, compute_gravity_resistance, compute_custom_metric"""
                        )
    parser.add_argument('-arm', '-a', type=str, default='left', help='Options: left, right.  Default: left'
                        )
    parser.add_argument('--baxter', action='store_true', help="""If you don\'t use this flag, you will only visualize the grasps.  This is
        so you can run this on your laptop"""
                        )
    parser.add_argument('--debug', action='store_true', help='Whether or not to use a random seed'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if True:#args.debug:
        np.random.seed(0)

    # Mesh loading and pre-processing
    mesh = trimesh.load_mesh('objects/{}.obj'.format(args.obj))
    T_obj_world = lookup_transform(args.obj)
    mesh.apply_transform(T_obj_world.matrix)
    mesh.fix_normals()


    # This policy takes a mesh and returns the best actions to execute on the robot
    grasping_policy = GraspingPolicy(
        args.n_vert,
        args.n_grasps,
        args.n_execute,
        args.n_facets,
        args.metric
    )
    # Each grasp is represented by T_grasp_world, a RigidTransform defining the
    # position of the end effector
    T_grasp_worlds = grasping_policy.top_n_actions(mesh, args.obj)

    # Execute each grasp on the baxter / sawyer
    if args.baxter:
        rospy.init_node('moveit_node')
        gripper = baxter_gripper.Gripper(args.arm)
        planner = PathPlanner('{}_arm'.format(args.arm))
        gripper.calibrate()

        for T_grasp_world in T_grasp_worlds:
            repeat = True
            while repeat:
                execute_grasp(T_grasp_world, planner, gripper)
                repeat = raw_input("repeat? [y|n] ") == 'y'
