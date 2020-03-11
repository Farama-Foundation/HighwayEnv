import numpy as np
import matplotlib.pyplot as plt

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
    FRICTION_FRONT = 10.0 * MASS  # [N]
    FRICTION_REAR = 10.0 * MASS  # [N]

    MAX_ANGULAR_VELOCITY = 2 * np.pi  # [rad/s]

    def __init__(self, road, position, heading=0, velocity=0):
        super().__init__(road, position, heading, velocity)
        self.lateral_velocity = 0
        self.yaw_rate = 0
        self.theta = None
        self.derivative = np.zeros(self.state.shape)
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

    @property
    def state(self):
        return np.array([[self.position[0]],
                         [self.position[1]],
                         [self.heading],
                         [self.velocity],
                         [self.lateral_velocity],
                         [self.yaw_rate]])

    def step(self, dt):
        self.clip_actions()

        x = np.array([[self.lateral_velocity], [self.yaw_rate]])
        u = np.array([[self.action['steering']]])
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        velocity = R @ np.array([self.velocity, self.lateral_velocity])
        self.derivative = np.array([[velocity[0]], [velocity[1]], [self.yaw_rate],
                                    [self.action['acceleration']], dx[0], dx[1]])

        self.position += velocity * dt
        self.heading += self.yaw_rate * dt
        self.velocity += self.action['acceleration'] * dt
        self.lateral_velocity += dx[0, 0] * dt
        self.yaw_rate += dx[1, 0] * dt

        self.on_state_update()

    def clip_actions(self):
        super().clip_actions()
        self.yaw_rate = np.clip(self.yaw_rate, -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)

    def lateral_lpv_structure(self):
        """
            State: [lateral velocity v, yaw rate r]
        :return: lateral dynamics dx = (A0 + theta^T phi)x + B u
        """
        speed_body_x = self.velocity
        A0 = np.array([
            [0, -speed_body_x],
            [0, 0]
        ])
        phi = np.array([
            [
                [-2 / (self.MASS*speed_body_x), -2*self.LENGTH_A / (self.MASS*speed_body_x)],
                [-2*self.LENGTH_A / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_A**2 / (self.INERTIA_Z*speed_body_x)]
            ], [
                [-2 / (self.MASS*speed_body_x), 2*self.LENGTH_B / (self.MASS*speed_body_x)],
                [2*self.LENGTH_B / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_B**2 / (self.INERTIA_Z*speed_body_x)]
            ],
        ])
        theta0 = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A0 += np.tensordot(theta0, phi, axes=[0, 0])
        B = np.array([
            [2*self.FRICTION_FRONT / self.MASS],
            [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z]
        ])
        return A0, phi, B

    def lateral_lpv_dynamics(self):
        """
            State: [lateral velocity v, yaw rate r]
        :return: lateral dynamics A, B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([0, 0])
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
        theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(theta, phi, axes=[0, 0])
        return A, np.array(B)


def simulate(dt=0.1):
    time = np.arange(0, 30, dt)
    vehicle = BicycleVehicle(road=None, position=[0, 5], velocity=5)
    xx, uu = [], []
    K = np.array([[1e-1, 2, 0, 1]])
    for t in time:
        # Act
        u = - K @ vehicle.state
        omega = 2*np.pi/20
        u += np.array([[-20*omega*np.sin(omega*t) * dt]])
        # Record
        xx.append(np.array([vehicle.position[0], vehicle.position[1], vehicle.heading])[:, np.newaxis])
        uu.append(u.copy())
        # Step
        vehicle.act({"acceleration": 0, "steering": u})
        vehicle.step(dt)
    xx, uu = np.array(xx), np.array(uu)
    plot(time, xx, uu)


def plot(time, xx, uu):
    pos_x = xx[:, 0, 0]
    pos_y = xx[:, 1, 0]
    psi_x = np.cos(xx[:, 2, 0])
    psi_y = np.sin(xx[:, 2, 0])
    dir_x = np.cos(xx[:, 2, 0] + uu[:, 0, 0])
    dir_y = np.sin(xx[:, 2, 0] + uu[:, 0, 0])
    fig, ax = plt.subplots(1, 1)
    # ax.plot(pos_x, pos_y)
    dir_scale = 1/5
    ax.quiver(pos_x[::20]-0.5/dir_scale*psi_x[::20],
              pos_y[::20]-0.5/dir_scale*psi_y[::20],
              psi_x[::20], psi_y[::20],
              angles='xy', scale_units='xy', scale=dir_scale, width=0.005, headwidth=1)
    ax.quiver(pos_x[::20]+0.5/dir_scale*psi_x[::20], pos_y[::20]+0.5/dir_scale*psi_y[::20], dir_x[::20], dir_y[::20],
              angles='xy', scale_units='xy', scale=0.25, width=0.005, color='r')
    ax.axis("equal")
    ax.grid()
    # ax1.plot(pos_x, xx[:, 3, 0])
    plt.show()
    plt.close()


def main():
    simulate()


if __name__ == '__main__':
    main()
