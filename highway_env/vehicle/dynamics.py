import numpy as np
import matplotlib.pyplot as plt

from highway_env.utils import not_zero
from highway_env.vehicle.kinematics import Vehicle


class BicycleVehicle(Vehicle):
    """
        This model is based on the following assumptions:
        - the vehicle is moving with a constant longitudinal speed
        - the steering input to front tires and the corresponding slip angles are small
        See https://pdfs.semanticscholar.org/bb9c/d2892e9327ec1ee647c30c320f2089b290c1.pdf, Chapter 3.
    """
    MASS = 1  # [kg]
    LENGTH_A = Vehicle.LENGTH / 2  # [m]
    LENGTH_B = Vehicle.LENGTH / 2  # [m]
    INERTIA_Z = 1/12 * MASS * (Vehicle.LENGTH ** 2 + 3 * Vehicle.WIDTH ** 2)  # [kg.m2]
    FRICTION_FRONT = 15.0 * MASS  # [N]
    FRICTION_REAR = 15.0 * MASS  # [N]

    MAX_ANGULAR_VELOCITY = 2 * np.pi  # [rad/s]
    MAX_VELOCITY = 15  # [m/s]

    def __init__(self, road, position, heading=0, velocity=0):
        super().__init__(road, position, heading, velocity)
        self.lateral_velocity = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

    @property
    def state(self):
        return np.array([[self.position[0]],
                         [self.position[1]],
                         [self.heading],
                         [self.velocity],
                         [self.lateral_velocity],
                         [self.yaw_rate]])

    @property
    def derivative(self):
        """
            See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)
        :return: the state derivative
        """
        delta_f = self.action["steering"]
        delta_r = 0
        theta_vf = np.arctan2(self.lateral_velocity + self.LENGTH_A * self.yaw_rate, self.velocity)  # (2.27)
        theta_vr = np.arctan2(self.lateral_velocity - self.LENGTH_B * self.yaw_rate, self.velocity)  # (2.28)
        f_yf = 2*self.FRICTION_FRONT * (delta_f - theta_vf)  # (2.25)
        f_yr = 2*self.FRICTION_REAR * (delta_r - theta_vr)  # (2.26)
        if abs(self.velocity) < 1:  # Low velocity dynamics: damping of lateral velocity and yaw rate
            f_yf = - self.MASS * self.lateral_velocity - self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
            f_yr = - self.MASS * self.lateral_velocity + self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
        d_lateral_velocity = 1/self.MASS * (f_yf + f_yr) - self.yaw_rate * self.velocity  # (2.21)
        d_yaw_rate = 1/self.INERTIA_Z * (self.LENGTH_A * f_yf - self.LENGTH_B * f_yr)  # (2.22)
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        velocity = R @ np.array([self.velocity, self.lateral_velocity])
        return np.array([[velocity[0]],
                         [velocity[1]],
                         [self.yaw_rate],
                         [self.action['acceleration']],
                         [d_lateral_velocity],
                         [d_yaw_rate]])

    @property
    def derivative_linear(self):
        x = np.array([[self.lateral_velocity], [self.yaw_rate]])
        u = np.array([[self.action['steering']]])
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        velocity = R @ np.array([self.velocity, self.lateral_velocity])
        return np.array([[velocity[0]], [velocity[1]], [self.yaw_rate], [self.action['acceleration']], dx[0], dx[1]])

    def step(self, dt):
        self.clip_actions()
        derivative = self.derivative
        self.position += derivative[0:2, 0] * dt
        self.heading += self.yaw_rate * dt
        self.velocity += self.action['acceleration'] * dt
        self.lateral_velocity += derivative[4, 0] * dt
        self.yaw_rate += derivative[5, 0] * dt

        self.on_state_update()

    def clip_actions(self):
        super().clip_actions()
        # Required because of the linearisation
        self.action["steering"] = np.clip(self.action["steering"], -np.pi/2, np.pi/2)
        self.yaw_rate = np.clip(self.yaw_rate, -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)

    def lateral_lpv_structure(self):
        """
            State: [lateral velocity v, yaw rate r]
        :return: lateral dynamics dx = (A0 + theta^T phi)x + B u
        """
        B = np.array([
            [2*self.FRICTION_FRONT / self.MASS],
            [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z]
        ])

        speed_body_x = self.velocity
        A0 = np.array([
            [0, -speed_body_x],
            [0, 0]
        ])

        if abs(speed_body_x) < 1:
            return A0, np.zeros((2, 2, 2)), B*0

        phi = np.array([
            [
                [-2 / (self.MASS*speed_body_x), -2*self.LENGTH_A / (self.MASS*speed_body_x)],
                [-2*self.LENGTH_A / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_A**2 / (self.INERTIA_Z*speed_body_x)]
            ], [
                [-2 / (self.MASS*speed_body_x), 2*self.LENGTH_B / (self.MASS*speed_body_x)],
                [2*self.LENGTH_B / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_B**2 / (self.INERTIA_Z*speed_body_x)]
            ],
        ])
        return A0, phi, B

    def lateral_lpv_dynamics(self):
        """
            State: [lateral velocity v, yaw rate r]
        :return: lateral dynamics A, B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([self.FRICTION_FRONT, self.FRICTION_REAR])
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

    def full_lateral_lpv_structure(self):
        """
            State: [position y, yaw psi, lateral velocity v, yaw rate r]
            The system is linearized around psi = 0
        :return: lateral dynamics A, B
        """
        A_lat, phi_lat, B_lat = self.lateral_lpv_structure()

        speed_body_x = self.velocity
        A_top = np.array([
            [0, speed_body_x, 1, 0],
            [0, 0, 0, 1]
        ])
        A0 = np.concatenate((A_top, np.concatenate((np.zeros((2, 2)), A_lat), axis=1)))
        phi = [np.concatenate((np.zeros((2, 4)), np.concatenate((np.zeros((2, 2)), phi_i), axis=1)))
               for phi_i in phi_lat]
        B = np.concatenate((np.zeros((2, 1)), B_lat))
        return A0, phi, B

    def full_lateral_lpv_dynamics(self):
        """
            State: [position y, yaw psi, lateral velocity v, yaw rate r]
            The system is linearized around psi = 0
        :return: lateral dynamics A, B
        """
        A0, phi, B = self.full_lateral_lpv_structure()
        self.theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B


def simulate(dt=0.1):
    import control
    time = np.arange(0, 20, dt)
    vehicle = BicycleVehicle(road=None, position=[0, 5], velocity=8.3)
    xx, uu = [], []
    from highway_env.interval import LPV
    A, B = vehicle.full_lateral_lpv_dynamics()
    K = -np.asarray(control.place(A, B, -np.arange(1, 5)))
    lpv = LPV(x0=vehicle.state[[1, 2, 4, 5]].squeeze(), a0=A, da=[np.zeros(A.shape)], b=B,
              d=[[0], [0], [0], [1]], omega_i=[[0], [0]], u=None, k=K, center=None, x_i=None)

    for t in time:
        # Act
        u = K @ vehicle.state[[1, 2, 4, 5]]
        omega = 2*np.pi/20
        u_p = 0*np.array([[-20*omega*np.sin(omega*t) * dt]])
        u += u_p
        # Record
        xx.append(np.array([vehicle.position[0], vehicle.position[1], vehicle.heading])[:, np.newaxis])
        uu.append(u)
        # Interval
        lpv.set_control(u, state=vehicle.state[[1, 2, 4, 5]])
        lpv.step(dt)
        x_i_t = lpv.change_coordinates(lpv.x_i_t, back=True, interval=True)
        # Step
        vehicle.act({"acceleration": 0, "steering": u})
        vehicle.step(dt)

    xx, uu = np.array(xx), np.array(uu)
    plot(time, xx, uu)


def plot(time, xx, uu):
    pos_x, pos_y = xx[:, 0, 0], xx[:, 1, 0]
    psi_x, psi_y = np.cos(xx[:, 2, 0]), np.sin(xx[:, 2, 0])
    dir_x, dir_y = np.cos(xx[:, 2, 0] + uu[:, 0, 0]), np.sin(xx[:, 2, 0] + uu[:, 0, 0])
    fig, ax = plt.subplots(1, 1)
    ax.plot(pos_x, pos_y, linewidth=0.5)
    dir_scale = 1/5
    ax.quiver(pos_x[::20]-0.5/dir_scale*psi_x[::20],
              pos_y[::20]-0.5/dir_scale*psi_y[::20],
              psi_x[::20], psi_y[::20],
              angles='xy', scale_units='xy', scale=dir_scale, width=0.005, headwidth=1)
    ax.quiver(pos_x[::20]+0.5/dir_scale*psi_x[::20], pos_y[::20]+0.5/dir_scale*psi_y[::20], dir_x[::20], dir_y[::20],
              angles='xy', scale_units='xy', scale=0.25, width=0.005, color='r')
    ax.axis("equal")
    ax.grid()
    plt.show()
    plt.close()


def main():
    simulate()


if __name__ == '__main__':
    main()
