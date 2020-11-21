#!/usr/bin/env python3
#
#   Kinematics Class
#
#   Use the KDL (Kinematics/Dynamics Library) to compute the forward
#   kinematics and Jacobian from a URDF.
#
import kdl_parser_py.urdf as kdlp
import PyKDL              as kdl


#
#  Kinematics Class Definition
#
class Kinematics:
    def __init__(self, urdf, world = 'world', tip = 'tip'):
        # Try loading the URDF data into a KDL tree.
        (ok, self.tree) = kdlp.treeFromString(urdf)
        assert ok, "Unable to parse the URDF"

        # Create the isolated chain from world to tip.
        self.chain = self.tree.getChain(world, tip)

        # Extract the number of joints.
        self.N = self.chain.getNrOfJoints()
        assert self.N>0, "Found no joints in the chain"

        # Create storage for the joint position, tip position, and
        # Jacobian matrices (for the KDL library).
        self.qkdl = kdl.JntArray(self.N)
        self.Tkdl = kdl.Frame()
        self.Jkdl = kdl.Jacobian(self.N)

        # Instantiate the solvers for tip position and Jacobian.
        self.fkinsolver = kdl.ChainFkSolverPos_recursive(self.chain)
        self.jacsolver  = kdl.ChainJntToJacSolver(self.chain)

    def dofs(self):
        # Return the number of joints.
        return self.N

    def fkin(self, q, p, R):
        # Load the KDL's joint position.
        for i in range(self.N):
            self.qkdl[i] = q[i]

        # Run the fkin solver.
        bad = self.fkinsolver.JntToCart(self.qkdl, self.Tkdl)
        assert not bad, "Forward Kinematics Solver failed"

        # Extract the position.
        for i in range(3):
            p[i] = self.Tkdl.p[i]

        # Extract the rotation.
        for i in range(3):
            for j in range(3):
                R[i,j] = self.Tkdl.M[i,j]

    def Jac(self, q, J):
        # Load the KDL's joint position.
        for i in range(self.N):
            self.qkdl[i] = q[i]

        # Run the fkin solver.
        bad = self.jacsolver.JntToJac(self.qkdl, self.Jkdl)
        assert not bad, "Jacobian Solver failed"

        # Extract the Jacobian matrix.
        for i in range(6):
            for j in range(self.N):
                J[i,j] = self.Jkdl[i,j]
