#!/usr/bin/env python3

import rospy
import tf as transforms
import numpy as np

from motions import Motion
from kinematics import Kinematics
from sensor_msgs.msg   import JointState

if __name__ == "__main__":
    # Prepare the node.
    rospy.init_node('motion')

    # Create a publisher to send the joint values (joint_states).
    # Note having a slightly larger queue prevents dropped messages!
    pub = rospy.Publisher("/joint_states", JointState, queue_size=100)

    # Wait until connected.  You don't have to wait, but the first
    # messages might go out before the connection and hence be lost.
    rospy.sleep(0.25)
    joints = {
        'back_bkx' : 0.0,
        'back_bky' : 0.22,
        'back_bkz' : 0.0,
        'l_arm_elx' : 1.0,
        'l_arm_ely' : 0.0,
        'l_arm_shx' : -1.5,
        'l_arm_shz' : -1.5,
        'l_arm_wrx' : 0.0,
        'l_arm_wry' : 0.0,
        'l_arm_wry2' : 0.0,
        'l_leg_akx' : 0.0,
        'l_leg_aky' : 0.0,
        'l_leg_hpx' : 0.0,
        'l_leg_hpy' : 0.0,
        'l_leg_hpz' : 0.0,
        'l_leg_kny' : 0.0,
        'neck_ry' : 0.0,
        'r_arm_elx' : -1.77,
        'r_arm_ely' : 0.31,
        'r_arm_shx' : -0.44,
        'r_arm_shz' : 1.57,
        'r_arm_wrx' : 0.0,
        'r_arm_wry' : 0.0,
        'r_arm_wry2' : 0.0,
        'r_leg_akx' : 0.0,
        'r_leg_aky' : 0.0,
        'r_leg_hpx' : 0.0,
        'r_leg_hpy' : 0.0,
        'r_leg_hpz' : 0.0,
        'r_leg_kny' : 0.0
    }

    # Grab the URDF from the parameter server.
    urdf = rospy.get_param('/robot_description')

    # Set up the kinematics, from pelvis to left and right foot.
    ll_kin = Kinematics(urdf, 'pelvis', 'l_foot')
    rl_kin = Kinematics(urdf, 'pelvis', 'r_foot')
    ll_N   = ll_kin.dofs()
    rl_N   = rl_kin.dofs()

    left_leg_joints_names = [
        'l_leg_hpz',
        'l_leg_hpx',
        'l_leg_hpy',
        'l_leg_kny',
        'l_leg_aky',
        'l_leg_akx'
    ]

    left_leg_q = np.array([[joints[n]] for n in left_leg_joints_names])

    right_leg_joints_names = [
        'r_leg_hpz',
        'r_leg_hpx',
        'r_leg_hpy',
        'r_leg_kny',
        'r_leg_aky',
        'r_leg_akx'
    ]

    right_leg_q = np.array([[joints[n]] for n in right_leg_joints_names])


    # Instantiate a broadcaster for
    broadcaster = transforms.TransformBroadcaster()

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))

    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    t   = 0.0
    tf  = 7.5
    lam = 0.1/dt
    switch_after_t = 0.75
    t_cycle = 0.0
    # this is the default, and means that the right leg is bent
    switched = False

    # Fun stuff for rotation and finite looping
    in_rotation = False
    rotate_after = 3.0

    reps = 0
    max_reps = 2

    motion = Motion(time_duration=switch_after_t)
    while not rospy.is_shutdown():
        # Move to a new time step, assuming a constant step!
        t = t + dt
        t_cycle = t_cycle + dt

        if t_cycle >= switch_after_t:
            t_cycle = 0.0

            if in_rotation:
                in_rotation = False
            else:
                switched = not switched

        elif (t >= rotate_after and t <= rotate_after + switch_after_t) or (t >= 2 * rotate_after + switch_after_t):
            in_rotation = True
            p_pw, quat_pw = motion.rotatePelvis(t)

        if not in_rotation:
            # Calculate the new position of the pelvis
            p_pw, quat_pw = motion.getPelvisPosition(t)

        # Here is where we will calculate the new joint values based on
        # time step and the current positions stored in joints dictionary

        # For our first example, we will try to move the left leg backwards
        # w.r.t the pelvis while keeping the foot parallel to the ground
            left_leg_q = motion.getJointAngles(ll_kin, t, left_leg_q, ll_N, dt, lam, straight=(not switched))
            for q_name, q in zip(left_leg_joints_names, left_leg_q):
                joints[q_name] = q

            right_leg_q = motion.getJointAngles(rl_kin, t, right_leg_q, rl_N, dt, lam, straight=switched, is_left=False)
            for q_name, q in zip(right_leg_joints_names, right_leg_q):
                joints[q_name] = q

        # create a joint state message
        msg = JointState()
        joint_values = []
        for joint_name, val in joints.items():
            msg.name.append(joint_name)
            joint_values.append(val)

        msg.position = joint_values

        # Send the command (with the current time) and sleep.
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)

        # Place the pelvis w.r.t. world.
        broadcaster.sendTransform(p_pw, quat_pw, rospy.Time.now(), 'pelvis', 'world')

        servo.sleep()

        if (t > tf):
            t = 0.0
            t_cycle = 0.0
            in_rotation = False
            switched = False
            reps += 1


        # Break if we have completed the full time.
        if (reps >= max_reps):
            break
