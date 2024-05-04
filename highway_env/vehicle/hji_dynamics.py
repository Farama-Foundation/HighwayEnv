#import heterocl as hcl
import numpy as np

class HJIVehicle:
    MAX_ACCELERATION: float = 5 # [m/s^2]
    MAX_STEERING_ANGLE: float = np.pi / 4 # [rad]

    def __init__(self, conservative=True, x_rel=0, y_rel=0, heading_rel=0, v_r=0, v_h = 0, uMode="max", dMode="min"):
        self.x_rel = x_rel
        self.y_rel = y_rel
        self.heading_rel = heading_rel
        self.v_r = v_r
        self.v_h = v_h
        self.uMode = uMode
        self.dMode = dMode
        self.conservative = conservative

    def opt_ctrl(self, t, state, spat_deriv):
        opt_a = hcl.scalar(self.MAX_ACCELERATION, "opt_a")
        opt_steer = hcl.scalar(self.MAX_STEERING_ANGLE, "opt_steer")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")

        au_term = hcl.scalar(0,"au_term")
        su_term = hcl.scalar(0, "su_term")
        au_term[0] = spat_deriv[3]
        su_term[0] = spat_deriv[2]

        with hcl.if_(au_term >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_a[0] = -opt_a
        with hcl.elif_(au_term < 0):
            with hcl.if_(self.uMode == "max"):
                opt_a[0] = -opt_a
        
        with hcl.if_(su_term >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_steer[0] = -opt_steer
        with hcl.elif_(su_term < 0):
            with hcl.if_(self.uMode == "max"):
                opt_steer[0] = -opt_steer
        return (opt_a[0], opt_steer[0], in3[0], in4[0], in5[0])

    def opt_dstb(self, t, state, spat_deriv):
        opt_a = hcl.scalar(self.MAX_ACCELERATION, "opt_a")
        opt_steer = hcl.scalar(self.MAX_STEERING_ANGLE, "opt_steer")
        # Just create and pass back, even though they're not used
        in3   = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")

        ad_term = hcl.scalar(0,"ad_term")
        sd_term = hcl.scalar(0, "sd_term")
        ad_term[0] = spat_deriv[4]
        sd_term[0] = spat_deriv[2]

        with hcl.if_(ad_term >= 0):
            with hcl.if_(self.dMode == "min"):
                opt_a[0] = -opt_a
        with hcl.elif_(ad_term < 0):
            with hcl.if_(self.dMode == "max"):
                opt_a[0] = -opt_a
        
        with hcl.if_(sd_term >= 0):
            with hcl.if_(self.dMode == "min"):
                opt_steer[0] = -opt_steer
        with hcl.elif_(sd_term < 0):
            with hcl.if_(self.dMode == "max"):
                opt_steer[0] = -opt_steer
        return (opt_a[0], opt_steer[0], in3[0], in4[0], in5[0])
    
    def tan_wrap(self, theta):
        return hcl.sin(theta)/hcl.cos(theta)
    
    def arctan_approx(self, x):
        return x - (x*x*x)/3 + (x*x*x*x*x*x)/5 + (x*x*x*x*x*x*x)/7
         
    def dynamics(self, t, state, uOpt, dOpt):
        print(f"start dynamics")
        x_rel_dot = hcl.scalar(0, "x_dot")
        y_rel_dot = hcl.scalar(0, "y_dot")
        heading_rel_dot = hcl.scalar(0, "heading_rel_dot")
        v_robot_dot = hcl.scalar(0, "v_robot")
        v_human_dot = hcl.scalar(0, "v_human")

        b_r = self.arctan_approx(1/2*self.tan_wrap(uOpt[1]))
        b_h = self.arctan_approx(1/2*self.tan_wrap(dOpt[1]))
        b_rel = b_h - b_r

        x_rel_dot[0] = self.v_r*hcl.cos(state[2]+b_rel) - self.v_h
        y_rel_dot[0] = self.v_h*hcl.sin(state[2]+b_rel)
        heading_rel_dot[0] = self.v_h/5*hcl.sin(b_h)-self.v_r/5*hcl.sin(b_r)
        v_robot_dot[0] = uOpt[0]
        v_human_dot[0] = dOpt[0]

        return (x_rel_dot[0], y_rel_dot[0], heading_rel_dot[0], v_robot_dot[0], v_human_dot[0])