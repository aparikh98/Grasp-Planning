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
OBJECT_H = {'gearbox': .056, 'nozzle': .032, 'pawn': .091}

def lookup_transform(to_frame, baxter, from_frame='base'):
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
    # if to_frame =='pawn':
    #     to_frame = 'ar_marker_9'
    if not baxter:
        print( 'I am the lookup transform function!  ' \
            + 'You\'re not using ROS, so I\'m returning the Identity Matrix.')
        tag_pos =   [0.642, -0.132, -0.023]
        tag_rot = [-0.001, -0.030, 0.024, 0.999]

        rot = RigidTransform.rotation_from_quaternion(tag_rot)
        return RigidTransform(rot, tag_pos, to_frame=from_frame, from_frame=to_frame)
        #return RigidTransform(translation = tag_pos, to_frame=from_frame, from_frame=to_frame)

    listener = tf.TransformListener()
    attempts, max_attempts, rate = 0, 10, rospy.Rate(1.0)
    while attempts < max_attempts:
        try:
            t = listener.getLatestCommonTime(from_frame, to_frame)
            tag_pos, tag_rot = listener.lookupTransform(
                from_frame, to_frame, t)
            break
        except:
            rate.sleep()
            attempts += 1
    # tag_pos =   [0.642, -0.132, -0.023]
    # tag_rot = [-0.001, -0.030, 0.024, 0.999]

    rot = RigidTransform.rotation_from_quaternion(tag_rot)
    return RigidTransform(rot, tag_pos, to_frame=from_frame, from_frame=to_frame)


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
        rospy.sleep(0.25)

    def open_gripper():
        """opens the gripper"""
        gripper.open(block=True)
        rospy.sleep(0.25)
    # final_position = np.asarray(T_grasp_world[0])
    # eucl_orientation = np.asarray(T_grasp_world[1])
    # print(eucl_orientation)
    # final_quaternion  = tfs.quaternion_from_euler(eucl_orientation[0], eucl_orientation[1], eucl_orientation[2])
    final_position = T_grasp_world.position
    intermediate_position = (T_grasp_world.position - np.reshape(np.matmul(T_grasp_world.rotation , np.array([[0], [0], [0.05]])), (1,3)))[0]
    final_position = (T_grasp_world.position + np.reshape(np.matmul(T_grasp_world.rotation , np.array([[0], [0], [0.02]])), (1,3)))[0]
   
    # final_quaternion = [-0.184, 0.981, -0.018, 0.065]
    # final_position = [0.7, -0.2, -0.2]

    final_quaternion = T_grasp_world.quaternion
    # eucl_orientation = tfs.euler_from_quaternion(final_quaternion)

    final_pose = PoseStamped()
    final_pose.header.frame_id = "base"
    final_pose.pose.position.x = final_position[0]
    final_pose.pose.position.y = final_position[1]
    final_pose.pose.position.z = final_position[2]
    final_pose.pose.orientation.x = final_quaternion[0]
    final_pose.pose.orientation.y = final_quaternion[1]
    final_pose.pose.orientation.z = final_quaternion[2]
    final_pose.pose.orientation.w = final_quaternion[3]
    print("Final Pose", final_pose)

    orien_const = OrientationConstraint()
    orien_const.link_name = "right_gripper";
    orien_const.header.frame_id = "base";
    orien_const.orientation = final_pose.pose.orientation
    orien_const.absolute_x_axis_tolerance = 0.1
    orien_const.absolute_y_axis_tolerance = 0.1
    orien_const.absolute_z_axis_tolerance = 0.1
    orien_const.weight = 1.0

    orien_const_ht = OrientationConstraint()
    orien_const_ht.link_name = "right_gripper";
    orien_const_ht.header.frame_id = "base";
    orien_const_ht.orientation = final_pose.pose.orientation
    orien_const_ht.absolute_x_axis_tolerance = 0.5
    orien_const_ht.absolute_y_axis_tolerance = 0.5
    orien_const_ht.absolute_z_axis_tolerance = 0.5
    orien_const_ht.weight = 1.0


    table_pose = PoseStamped()
    table_pose.header.frame_id = "base"
    table_pose.pose.position.x = 0
    table_pose.pose.position.y = 0
    table_pose.pose.position.z = -1
    table_pose.pose.orientation.x = 0
    table_pose.pose.orientation.y = 0
    table_pose.pose.orientation.z = 0
    table_pose.pose.orientation.w = 1

    planner.add_box_obstacle(np.array([2, 2, 2* abs(-1-table_h) + 0.04]), "table", table_pose)

   
    intermediate_pose = PoseStamped()
    intermediate_pose.header.frame_id = "base"
    intermediate_pose.pose.position.x = intermediate_position[0]
    intermediate_pose.pose.position.y = intermediate_position[1]
    intermediate_pose.pose.position.z = intermediate_position[2]
    intermediate_pose.pose.orientation.x = final_quaternion[0]
    intermediate_pose.pose.orientation.y = final_quaternion[1]
    intermediate_pose.pose.orientation.z = final_quaternion[2]
    intermediate_pose.pose.orientation.w = final_quaternion[3]

    above_pose = PoseStamped()
    above_pose.header.frame_id = "base"
    above_pose.pose.position.x = final_position[0]
    above_pose.pose.position.y = final_position[1]
    above_pose.pose.position.z = final_position[2] + 0.1
    above_pose.pose.orientation.x = final_quaternion[0]
    above_pose.pose.orientation.y = final_quaternion[1]
    above_pose.pose.orientation.z = final_quaternion[2]
    above_pose.pose.orientation.w = final_quaternion[3]

    place_pose = PoseStamped()
    place_pose.header.frame_id = "base"
    place_pose.pose.position.x = final_position[0] + 0.05
    place_pose.pose.position.y = final_position[1] + 0.05
    place_pose.pose.position.z = final_position[2]
    place_pose.pose.orientation.x = final_quaternion[0]
    place_pose.pose.orientation.y = final_quaternion[1]
    place_pose.pose.orientation.z = final_quaternion[2]
    place_pose.pose.orientation.w = final_quaternion[3]


    print("intermediate Pose", intermediate_pose)
    # while not rospy.is_shutdown():
    #     try:
    #         plan = planner.plan_to_pose(intermediate_pose, list())
    #         inp = raw_input('Press <Enter> to move to intermediate_pose, or \'exit\' to exit')
    #         # print(plan)
    #         if not plan or not planner.execute_plan(plan):
    #             raise Exception("Execution failed retrying")
    #         else:
    #             break
    #     except Exception as e:
    #         print e
    #         plan = planner.plan_to_pose(intermediate_pose, list())
    #         inp = raw_input('Press <Enter> to move to intermediate_pose, or \'exit\' to exit')

    # inp = raw_input('Press <Enter> to open gripper, or \'exit\' to exit')
    # open_gripper()

    plan = planner.plan_to_pose(intermediate_pose, list())
    inp = raw_input('Press <Enter> to move to intermediate_pose, or \'exit\' to exit')
    if (inp == 'exit'):
        return
    planner.execute_plan(plan)

    plan2 = planner.plan_to_pose(final_pose, [orien_const])
    inp = raw_input('Press <Enter> to move to final pose, or \'exit\' to exit')
    if (inp == 'exit'):
        return

    planner.execute_plan(plan2)
    
    inp = raw_input('Press <Enter> to close gripper, or \'exit\' to exit')
    if (inp == 'exit'):
        return

    close_gripper()

    plan = planner.plan_to_pose(above_pose, [orien_const])
    inp = raw_input('Press <Enter> to move to above pose, or \'exit\' to exit')
    planner.execute_plan(plan)

    plan = planner.plan_to_pose(place_pose, [orien_const])
    inp = raw_input('Press <Enter> to move to placepose, or \'exit\' to exit')
    planner.execute_plan(plan)


    inp = raw_input('Press <Enter> to open gripper, or \'exit\' to exit')
    open_gripper()


    # while not rospy.is_shutdown():
    #     try:
    #         plan2 = planner.plan_to_pose(final_pose, [orien_const])
    #         inp = raw_input('Press <Enter> to move to final pose, or \'exit\' to exit')
    #         # print(plan)
    #         if not plan2 or not planner.execute_plan(plan2):
    #             raise Exception("Execution failed retrying")
    #         else:
    #             break
    #     except Exception as e:
    #         print e

    # while not rospy.is_shutdown():
    #     try:
    #         plan = planner.plan_to_pose(intermediate_pose, [orien_const])
    #         inp = raw_input('Press <Enter> to move to intermediate_pose, or \'exit\' to exit')
    #         # print(plan)
    #         if not plan or not planner.execute_plan(plan):
    #             raise Exception("Execution failed retrying")
    #         else:
    #             break
    #     except Exception as e:
    #         print e

    if inp == "exit":
        return


def parse_args():
    """
    Pretty self explanatory tbh
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', type=str, default='pawn', help="""Which Object you\'re trying to pick up.  Options: gearbox, nozzle, pawn.
        Default: pawn"""
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
    parser.add_argument('-arm', '-a', type=str, default='right', help='Options: left, right.  Default: right'
                        )
    parser.add_argument('--baxter', action='store_true', help="""If you don\'t use this flag, you will only visualize the grasps.  This is
        so you can run this on your laptop"""
                        )
    parser.add_argument('--debug', action='store_true', help='Whether or not to use a random seed'
                        )
    parser.add_argument('-num_exec', '-n', type=int, default='0', help='Options: 0, 1,..top_n_actions.  Default: 0'
                    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.baxter:
        rospy.init_node('lab2_node')

    if args.debug:
        np.random.seed(0)

    # Mesh loading and pre-processing
    mesh = trimesh.load_mesh('objects/{}.obj'.format(args.obj))
    T_obj_world = lookup_transform(args.obj, args.baxter)
    mesh.apply_transform(T_obj_world.matrix)
    mesh.fix_normals()
    
    table_h = mesh.center_mass[2] -OBJECT_H[args.obj]

    repeat =True
    if args.baxter:
        gripper = baxter_gripper.Gripper(args.arm)
        planner = PathPlanner('{}_arm'.format(args.arm))
        gripper.calibrate()
    grasping_policy = GraspingPolicy(
        args.n_vert,
        args.n_grasps,
        args.n_execute,
        args.n_facets,
        args.metric
    )
    if args.baxter:

        while repeat:

        # This policy takes a mesh and returns the best actions to execute on the robot
            # Each grasp is represented by T_grasp_world, a RigidTransform defining the
            # position of the end effector
            T_grasp_worlds = grasping_policy.top_n_actions(mesh, args.obj)

            n = args.num_exec
            # Execute each grasp on the baxter / sawyer
            repeat2 = True
            while repeat2:
                gripper = baxter_gripper.Gripper(args.arm)
                planner = PathPlanner('{}_arm'.format(args.arm))
                gripper.calibrate()
                T_grasp_world = T_grasp_worlds[n]
                execute_grasp(T_grasp_world, planner, gripper)
                repeat2 = raw_input("repeat? [y|n] ") == 'y'
            repeat = raw_input("repeat calculation? [y|n] ") == 'y'

