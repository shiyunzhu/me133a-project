#!/usr/bin/env python3

import rospy
import tf as transforms
import numpy as np

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
        'back_bky' : 0.0,
        'back_bkz' : 0.0,
        'l_arm_elx' : 0.0,
        'l_arm_ely' : 0.0,
        'l_arm_shx' : 0.0,
        'l_arm_shz' : 0.0,
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
        'r_arm_elx' : 0.0,
        'r_arm_ely' : 0.0,
        'r_arm_shx' : 0.0,
        'r_arm_shz' : 0.0,
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

    # Prepare a servo loop at 100Hz.
    rate  = 100;
    servo = rospy.Rate(rate)
    dt    = servo.sleep_dur.to_sec()
    rospy.loginfo("Running the servo loop with dt of %f seconds (%fHz)" %
                  (dt, rate))

    # Run the servo loop until shutdown (killed or ctrl-C'ed).
    t   = 0.0
    tf  = 2.0
    lam = 0.1/dt
    while not rospy.is_shutdown():
        # Move to a new time step, assuming a constant step!
        t = t + dt

        # Here is where we will calculate the new joint values based on
        # time step and the current positions stored in joints dictionary
        joints['r_leg_kny'] = np.pi / 4
        joints['l_leg_kny'] = np.pi / 4

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
        servo.sleep()

        # Break if we have completed the full time.
        if (t > tf):
            break
