#!/usr/bin/env python3
import numpy as np
import tf as transforms

from kinematics import Kinematics
import utils

class Motion():

    def __init__(self, leg_length=0.8620):
        self.leg_length = leg_length
        self.leg_angle = 0.0
        self.s = 0.0
        self.sdot = 0.0

    def _update_s(self, t):
        self.s = 0
        self.sdot = 0

    def _update_leg_angle(self, t):
        self.leg_angle = (20 - 40 * (t % 1)) * np.pi / 180

    # ==========================================================================
    # Functions for straight leg movement
    # ==========================================================================
    def _get_straight_leg_velocity(self):
        xdot = -(np.pi / 6.0) * np.cos(self.leg_angle) * self.leg_length
        ydot = 0
        zdot = -(np.pi / 6.0) * np.sin(self.leg_angle) * self.leg_length
        return np.array([[xdot],[ydot],[zdot]])

    def _get_straight_leg_position(self):
        x = np.sin(self.leg_angle) * self.leg_length
        y = 0.114083
        z = -np.cos(self.leg_angle) * self.leg_length
        return np.array([[x],[y],[z]])

    def _get_straight_leg_R(self):
        return np.identity(3)

    def _get_straight_leg_omega(self):
        return np.array([[0], [0], [0]])

    # ==========================================================================
    # Functions for bent leg motion
    # ==========================================================================
    def _get_bent_leg_position(self):
        foot_length = 0.1
        x = np.sin(-self.leg_angle) * self.leg_length
        y = 0.114083
        z = -np.cos(-self.leg_angle) * self.leg_length + foot_length
        return np.array([[x],[y],[z]])

    def _get_bent_leg_velocity(self):
        xdot = (np.pi / 6.0) * np.cos(-self.leg_angle) * self.leg_length
        ydot = 0
        zdot = (np.pi / 6.0) * np.sin(-self.leg_angle) * self.leg_length
        return np.array([[xdot],[ydot],[zdot]])

    def _get_bent_leg_R(self):
        return utils.Ry(np.pi / 4)

    def _get_bent_leg_omega(self):
        return np.array([[0], [0], [0]])

    # ==========================================================================
    # Public functions
    # ==========================================================================
    def getPelvisPosition(self, t):
        # Calculate the new position of the pelvis
        x = -0.5 * t
        y = 0.0
        z = np.cos(self.leg_angle) * self.leg_length

        p_pw = np.array([[x],[y],[z]])# Choose the pelvis w.r.t. world position
        R_pw = np.identity(3)     # Choose the pelviw w.r.t. world orientation
        # Determine the quaternions for the orientation, using a T matrix:
        T_pw = np.vstack((np.hstack((R_pw, p_pw)),np.array([[0, 0, 0, 1]])))
        quat_pw = transforms.transformations.quaternion_from_matrix(T_pw)
        return p_pw, quat_pw

    def getJointAngles(self, kin, t, q, N, dt, lam, straight=True):
        # update leg_angle
        self._update_s(t)
        self._update_leg_angle(t)

        J = np.zeros((6,N))
        p = np.zeros((3,1))
        R = np.identity(3)

        # this changes the position and R
        kin.fkin(q, p, R)

        # this calculates the Jacobian
        kin.Jac(q, J)

        if straight:
            R_d = self._get_straight_leg_R()
            omega = self._get_straight_leg_omega()
            p_dot = self._get_straight_leg_velocity()
            x = self._get_straight_leg_position()
        else:
            R_d = self._get_bent_leg_R()
            omega = self._get_bent_leg_omega()
            p_dot = self._get_bent_leg_velocity()
            x = self._get_bent_leg_position()


        velocity = np.vstack((p_dot, omega))
        error_p = utils.get_error_p(x, p)
        error_r = utils.get_error_r(R, R_d)
        error = np.vstack((error_p, error_r))
        vr = velocity + lam * error

        qdot = np.linalg.pinv(J, 0.001) @ vr
        q = q + qdot * dt

        return q
