import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from pathos import multiprocessing
from typing import Tuple, List, Union, Optional
import math
import os
from tqdm import tqdm
import warnings
from slugify import slugify
import multiprocessing


class DefinitelyHumanHead:
    def __init__(self, radius=10.0, f_g=0.1, f_x=0.0,
                 n_processes=3, use_phi=False):
        """
        Initialise a human head object for hair simulations.

        :param radius: Radius of the human head (units: cm).
            Once set can not be changed within the same object.
        :param f_g: Force of gravity relative to bending stiffness
            of hair (units: cm^(-3))
        :param f_x: Force of wind acting in +x direction relative
            to bending stiffness of hair (units: cm^(-3))
        :param n_processes: Number of python processes to use for
            simulating, default 3, if this runs inordinately slowly
            or makes your computer unusable, reduce it. If you feel
            adventurous, increase it.
        :param use_phi: Whether or not to generate a 3d head
            environment for simulation, constraints apply to certain
            method arguments when in 2d mode.

        :returns: self
        :rtype: DefinitelyHumanHead
        """
        self.__head_radius = radius
        self.f_g = f_g
        self.f_x = f_x
        self.max_loops = 100
        self.use_phi = use_phi
        self.convergence_mag = 1.0e-3
        self.hairs_polar = []
        self.hairs_cartesian = []
        self.simulated_hairs = []
        if not use_phi:
            self.fig, self.ax = plt.subplots()
        else:
            pass
        self.step_dist = 0
        self.n_finite_elements = 0

        self.n_processes = n_processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_head_radius(self):
        return self.__head_radius

    def gen_hair_root(self, root_theta, root_phi=0.0, length=4.0):
        """This method when considered part of the class satisfies
        the requirements of question 1, short of parsing a list of
        angles. Generate hair root lists externally and loop through
        them to add hair roots for simulation.

        :param root_theta: Latitude, angle away from horizontal,
            radians.
        :type root_theta: float
        :param root_phi: Longitude, angle away from x-axis about z-axis,
            radians.
        :param length: Length of the hair when it is to be simulated,
            centimeters.
        """
        if root_phi != 0.0 and not self.use_phi:
            print("root_phi was passed when phi is not in use for "
                  "this simulation. Reinitialise the class with "
                  "use_phi=True to use phi angles in hair generation.")
            return
        self.hairs_polar.append([root_theta, root_phi, length])
        self.hairs_cartesian.append(
            list(self.__root_to_cart(root_theta, root_phi)).append(length)
        )

    def __hair_simulator(self, i):
        """
        Internal wrapping class-method, do not call directly.

        Does not do much beyond formatting inputs properly for other methods.
        :param i: Hair identification index.
        :return: Simulated hair cartesian locations.
        """
        # First generate a guess for each hair of a straight
        # strand perpendicular to the head.
        # Two elements are added to the requested number to
        # account for handling boundary conditions.
        self.step_dist = self.hairs_polar[i][2] / (self.n_finite_elements - 1)
        if not self.use_phi:
            hair_guess = (
                np.full(self.n_finite_elements, self.hairs_polar[i][0]),
                np.zeros_like(self.n_finite_elements))
            return self.__relax_hair_no_phi(hair_guess, self.hairs_polar[i][2])
        else:
            hair_guess = (
                np.full(self.n_finite_elements, self.hairs_polar[i][0]),
                np.full(self.n_finite_elements, self.hairs_polar[i][1]))
            return self.__relax_hair_w_phi(hair_guess)

    def simulate_hairs(self, n_finite_elements=60, plot_title=""):
        """
        Single call class-method that will begin the simulations of all hairs on
        the head (i.e. within self.hairs_polar) according to environment
        parameters like self.f_g and self.f_x which are class attributes defined
        on initialisation.

        This method does not return anything, to access simulated hairs once this
        is complete, find hairs in self.simulated_hairs, a list of hairs, where
        each element is a list of [x, y, z] lists containing the respective hair's
        cartesian locations.

        :param n_finite_elements: Number of internal finite elements to simulate
            hairs with, increasing this number will make the simulation slower.
        :param plot_title: In the case self.use_phi=False this will be the title of
            automatically generated 2d-plots, these plots will also be given a
            similar file name. If left blank, no internal plotting will be done.
        """
        self.n_finite_elements = n_finite_elements + 1
        # Reset the list of simulated hairs and the plot we output to
        self.simulated_hairs = []
        make_plot = False

        if not self.use_phi and plot_title != "":
            make_plot = True
            self.fig, self.ax = plt.subplots()
            self.fig.constrained = False
            self.__plot_head_2d()
            self.ax.set_xlabel("x / cm")
            self.ax.set_ylabel("z / cm")
            self.ax.set_aspect("equal")

        loc_worker_pool = multiprocessing.Pool(self.n_processes)

        # Create our mapped tasks for the worker pool.
        abs_coord_results = loc_worker_pool.imap_unordered(
            self.__hair_simulator,
            range(len(self.hairs_polar)))

        # As worker pool results roll in, process them and use the
        # generated tqdm progress bar to give user feedback.
        for result in tqdm(abs_coord_results, total=len(self.hairs_polar),
                           dynamic_ncols=True, ascii=True, unit="hair"):
            self.simulated_hairs.append(result)
            if make_plot:
                self.ax.plot(*result, color="black")

        # Close the worker pool so to avoid leaving dead python processes
        # all over the place.
        loc_worker_pool.close()

        if make_plot:
            if not os.path.exists("./fwf-plots/"):
                os.mkdir("./fwf-plots/")
            self.ax.set_title(plot_title)
            self.fig.savefig(f"./fwf-plots/{slugify(plot_title)}.png", dpi=300)

    def __relax_hair_w_phi(self, hair_guess_0):
        """Internal method, do not call directly.

        This method uses finite difference relaxation to find the rest
        positions of hairs within this simulation object.

        Relaxation was used due to instabilities found when f_x != 0
        when using the shooting method as well as the relative ease of
        implementation when compared to other, more compute efficient
        algorithms like collocation.
        """
        def func_a_rhs(s_dist, theta_loc, phi_loc):
            return s_dist * self.f_g * math.cos(theta_loc) + \
                   s_dist * self.f_x * math.sin(theta_loc) * math.cos(phi_loc)

        def func_b_rhs(s_dist, theta_loc, phi_loc):
            return -s_dist * self.f_x * math.sin(phi_loc) * math.sin(theta_loc)

        def func_a_diff_wrt_theta(s_dist, theta_loc, phi_loc):
            return (-s_dist * self.f_g * math.sin(theta_loc) +
                    s_dist * self.f_x * math.cos(phi_loc) * math.sin(phi_loc))

        def func_b_diff_wrt_phi(s_dist, theta_loc, phi_loc):
            return (-s_dist * self.f_x * math.cos(phi_loc) *
                    math.sin(theta_loc))

        def gen_jacobian(derivative_func, theta_list, phi_list):
            """Jacobian generation for this system
            """
            middle_diag = []
            for i in range(1, len(theta_list)):
                middle_diag.append(-2 - self.step_dist ** 2 *
                                   derivative_func(self.step_dist, theta_list[i],
                                                   phi_list[i]))
            upper_diag = np.full(len(middle_diag) - 1, 1.0)
            lower_diag = upper_diag
            jacobian_matrix = sparse.diags([upper_diag, middle_diag,
                                            lower_diag], offsets=[1, 0, -1])
            # print(jacobian_matrix.toarray().shape)
            return jacobian_matrix.toarray()

        def gen_residual(rhs_func, theta_list, phi_list, use_arg):
            """Generation of the residual of the system, in this process,
            adds another ghost point onto the end of the set of angle values
            to allow for residual generation for the ghost value.
            """
            theta_list = np.append(theta_list, theta_list[-1])
            phi_list = np.append(phi_list, phi_list[-1])
            residual_vec = np.zeros_like(theta_list)
            if use_arg == "theta_loc":
                for i in range(1, len(residual_vec) - 1):
                    residual_vec[i] = theta_list[i - 1] + theta_list[i + 1] - \
                                      2 * theta_list[i] - self.step_dist ** 2 \
                                      * rhs_func(self.step_dist * i,
                                                 theta_list[i], phi_list[i])
            elif use_arg == "phi_loc":
                phi_list = np.append(phi_list, phi_list[-1])
                for i in range(1, len(residual_vec) - 1):
                    residual_vec[i] = phi_list[i - 1] + phi_list[i + 1] - \
                                      2 * phi_list[i] - self.step_dist ** 2 \
                                      * rhs_func(self.step_dist * i,
                                                 theta_list[i], phi_list[i])
            # print(len(residual_vec[1:-1]))
            return residual_vec[1:-1]

        # Add the ghost point to the end of the angle lists
        guess_theta = np.append(hair_guess_0[0], hair_guess_0[0][-1])
        guess_phi = np.append(hair_guess_0[1], hair_guess_0[1][-1])

        # Generate dummy correction vectors to be changed later
        correction_theta = np.full(len(guess_theta), 1)
        correction_phi = correction_theta.copy()

        loop_count = 0
        while np.linalg.norm(correction_theta[:-2] + correction_phi[:-2]) > \
                self.convergence_mag and loop_count < self.max_loops:
            loop_count += 1
            # Generate the new correction vector for theta_loc and add it to theta_loc
            correction_theta = np.linalg.solve(
                gen_jacobian(func_a_diff_wrt_theta, guess_theta, guess_phi),
                -1 * gen_residual(func_a_rhs, guess_theta, guess_phi, "theta_loc"))
            guess_theta = np.add(guess_theta, [0.0, *correction_theta])
            # Enforce boundary conditions
            guess_theta[0] = hair_guess_0[0][0]
            guess_theta[-1] = guess_theta[-2] = guess_theta[-3]
            # Generate a new correction vector for phi_loc using the better theta_loc
            # guesstimate.
            correction_phi = np.linalg.solve(
                gen_jacobian(func_b_diff_wrt_phi, guess_theta, guess_phi),
                -1 * gen_residual(func_b_rhs, guess_theta, guess_phi, "phi_loc")
            )

            guess_phi = np.add(guess_phi, [0.0, *correction_phi])
            guess_phi[0] = hair_guess_0[1][0]
            guess_phi[-1] = guess_phi[-2] = guess_phi[-3]
            # print(np.linalg.norm(correction_theta[:-2] + correction_phi[:-2]))

        final_hair_angles_theta = guess_theta[0:-1]
        final_hair_angles_phi = guess_phi[0:-1]

        return self.__convert_hair_to_cartesian(
            (final_hair_angles_theta, final_hair_angles_phi),
            self.step_dist, (hair_guess_0[0][0], hair_guess_0[1][0]))

    def __relax_hair_no_phi(self, hair_guess_0, hair_length):
        """Internal method, do not call directly.

        This method uses finite difference relaxation to find the rest
        positions of hairs within this simulation object.

        Relaxation was used due to instabilities found when f_x != 0
        when using the shooting method as well as the relative ease of
        implementation when compared to other, more compute efficient
        algorithms like collocation.
        """

        # Boundary conditions in this case are that the ghost point we
        # create beyond the final element of hair must be equal to the real
        # final finite element of hair and that the the root element must
        # be fixed as the root polar location.

        def func_rhs_no_phi(s_dist, theta_loc):
            return s_dist * self.f_g * math.cos(theta_loc) + \
                   s_dist * self.f_x * math.sin(theta_loc)

        def func_rhs_diff_wrt_theta_no_phi(s_dist, theta_loc):
            return -s_dist * self.f_g * math.sin(theta_loc) + \
                   s_dist * self.f_x * math.cos(theta_loc)

        def gen_residual(hair_sol_guess):
            """Function that will generate a residual vector from theta_loc guess.
            An extra ghost point is added on here to allow for residual
            generation for the ghost point.
            """
            hair_sol_guess = np.append(hair_sol_guess, hair_sol_guess[-1])
            residual_vec = np.zeros_like(hair_sol_guess)
            for i in range(1, len(hair_sol_guess) - 1):
                residual_vec[i] = \
                    hair_sol_guess[i - 1] + hair_sol_guess[i + 1] - \
                    2 * hair_sol_guess[i] - \
                    self.step_dist ** 2 * func_rhs_no_phi(self.step_dist * i,
                                                          hair_sol_guess[i])

            return residual_vec[1:-1]

        def gen_jacobian(hair_sol_guess):
            """Function that will generate jacobian matrix from theta_loc guess.
            """
            middle_diag = []
            for i in range(1, len(hair_sol_guess)):
                middle_diag.append(
                    -2 - self.step_dist ** 2 * func_rhs_diff_wrt_theta_no_phi(
                        self.step_dist * i, hair_sol_guess[i])
                )
            upper_diag = np.full(len(middle_diag) - 1, 1.0)
            lower_diag = upper_diag
            jacobian = sparse.diags([upper_diag, middle_diag, lower_diag],
                                    offsets=[1, 0, -1]).toarray()
            return jacobian

        # Appending a ghost element to the end of the hair to consider
        # the Neumann boundary condition.
        self.step_dist = hair_length / (len(hair_guess_0[0]) - 1)
        hair_guess = np.append(hair_guess_0[0], hair_guess_0[0][-1])
        correction_vector = np.full(len(hair_guess), 10.0)
        loop_count = 0
        while np.linalg.norm(correction_vector[:-2]) > self.convergence_mag \
                and loop_count < self.max_loops:
            loop_count += 1
            # Loop-wise enforcement of boundary conditions.
            hair_guess[0] = hair_guess_0[0][0]
            hair_guess[-1] = hair_guess[-2] = hair_guess[-3]
            # Generation and alignment of the correction to be applied to
            # the guess.
            correction_vector = np.linalg.solve(gen_jacobian(hair_guess),
                                                -1 * gen_residual(hair_guess))
            # Making correction vector map to the angle vector before adding
            # the two.
            correction_vector = [0.0, *correction_vector]
            # print(np.linalg.norm(correction_vector))
            hair_guess = np.add(hair_guess, correction_vector)

        # Assignment of the final, satisfactorily simulated hair before its
        # conversion to cartesian and return up the function chain as a result
        final_hair_angles_theta = hair_guess
        # print(final_hair_angles_theta)
        hair_cartesian_locations = self.__convert_hair_to_cartesian(
            final_hair_angles_theta[:-1],
            self.step_dist, hair_guess_0[0][-1])

        # We return only the x and z absolute locations.
        return hair_cartesian_locations[0::2]

    def __convert_hair_to_cartesian(self, hair_angles, step_dist, hair_root) -> \
            List[List[List[float]]]:
        """Discrete conversion of hair from polar finite elements to absolute
        cartesian coordinates.

        :param hair_angles: Angles of each finite element of hair, type depends
            on self.use_phi.
        :type hair_angles: Union[List[float], Tuple[List[float], List[float]]]
        :param step_dist: Length of each finite element
        :type step_dist: float
        :param hair_root: Hair root location on scalp of head given in spherical
            polar coordinate angles.
        :type hair_root: Union[float, Tuple[float, float]]
        :return: List of x, y, z lists of float values for absolute locations of
            each finite element of the hair.
        """

        if not self.use_phi:
            # The 2d case, generate hair root absolute locations
            x_offset_list = [self.__head_radius * np.cos(hair_root)]
            z_offset_list = [self.__head_radius * np.sin(hair_root)]
            # For each element of hair, append a cartesian offset.
            x_offset_list = np.append(x_offset_list,
                                      step_dist * np.cos(hair_angles))
            z_offset_list = np.append(z_offset_list,
                                      step_dist * np.sin(hair_angles))
            y_offset_list = np.zeros_like(x_offset_list)
        else:
            # The 3d case, again generate root locations.
            x_offset_list, y_offset_list, z_offset_list = \
                self.__root_to_cart(*[angle[0] for angle in hair_angles])
            # Append the list of cartesian offsets in each dimension
            x_offset_list = np.append(x_offset_list,
                                      step_dist * np.cos(hair_angles[0]) *
                                      np.cos(hair_angles[1]))
            y_offset_list = np.append(y_offset_list,
                                      step_dist * -np.cos(hair_angles[0]) *
                                      np.sin(hair_angles[1]))
            z_offset_list = np.append(z_offset_list,
                                      step_dist * np.sin(hair_angles[0]))

        # Cumulative summation to get the true values from offsets.
        x_abs_list = np.cumsum(x_offset_list)
        y_abs_list = np.cumsum(y_offset_list)
        z_abs_list = np.cumsum(z_offset_list)

        return [x_abs_list, y_abs_list, z_abs_list]

    def __root_to_cart(self, root_theta, root_phi) -> \
            Tuple[float, float, float]:
        """
        Internal method, do not call directly.

        Converts root polar locations to cartesian.

        :param root_theta: Hair root latitude, radians
        :type root_theta:
        :param root_phi: Hair root longitude, radians
        :return: (x location, y location, z location)
        """
        x_loc = self.__head_radius * math.cos(root_theta) * \
            math.cos(root_phi)
        y_loc = -self.__head_radius * math.cos(root_theta) * \
            math.sin(root_phi)
        z_loc = self.__head_radius * math.sin(root_theta)
        return x_loc, y_loc, z_loc

    def __plot_head_2d(self):
        """
        Internal method used for internal plotting, generates
        the circle used to show the head in 2d plots."""
        circle_angle = np.linspace(0.0, 2 * math.pi, 200)
        x_vals = self.__head_radius * np.cos(circle_angle)
        z_vals = self.__head_radius * np.sin(circle_angle)
        self.ax.plot(x_vals, z_vals)


# This is here because matplotlib was behaving oddly and throwing
# warnings for no discernible reason. If debugging, change this.
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Test cases, usage patterns documented above and in run.py

    with DefinitelyHumanHead() as test_head:
        for theta in np.linspace(0, 2*np.pi, num=10):
            test_head.gen_hair_root(theta)
        test_head.simulate_hairs(plot_title="Head Test Plot, no wind.")
        test_head.f_x = 0.1
        test_head.simulate_hairs(plot_title="Head Test Plot, f_x = 0.1")
    with DefinitelyHumanHead(use_phi=True) as test_head_3d:
        test_head_3d.convergence_mag = 1e-3
        theta, phi = np.meshgrid(np.linspace(0.0, 0.49 * np.pi, num=5),
                                 np.linspace(0.0, np.pi, num=5))
        theta_phi_combinations = np.vstack([theta.ravel(), phi.ravel()]).T
        for hair_root_loc in theta_phi_combinations:
            test_head_3d.gen_hair_root(*hair_root_loc)
        test_head_3d.simulate_hairs(n_finite_elements=40)
        # noinspection DuplicatedCode
        figs_2d, (ax_2d_x_z, ax_2d_y_z) = plt.subplots(1, 2)
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        phi_angle = np.linspace(0.0, 2 * math.pi, 100)
        theta_angle = np.linspace(0.0, math.pi, 25)
        sphere_x = test_head_3d.get_head_radius() * np.outer(np.cos(phi_angle),
                                                             np.sin(theta_angle))
        sphere_y = test_head_3d.get_head_radius() * np.outer(np.sin(phi_angle),
                                                             np.sin(theta_angle))
        sphere_z = test_head_3d.get_head_radius() * np.outer(np.ones(np.size(phi_angle)),
                                                             np.cos(theta_angle))
        ax_3d.plot_surface(sphere_x, sphere_y, sphere_z, color="yellow")
        x_vals_l = test_head_3d.get_head_radius() * np.cos(phi_angle)
        z_vals_l = test_head_3d.get_head_radius() * np.sin(phi_angle)
        ax_2d_x_z.plot(x_vals_l, z_vals_l, color="blue")
        ax_2d_y_z.plot(x_vals_l, z_vals_l, color="blue")
        ax_2d_x_z.set_aspect("equal")
        ax_2d_y_z.set_aspect("equal")
        for hair in test_head_3d.simulated_hairs:
            ax_2d_x_z.plot(hair[0], hair[2], color="black")
            ax_2d_y_z.plot(hair[1], hair[2], color="black")
            ax_3d.plot(*hair, color="black", zorder=5)

        ax_2d_x_z.set_xlabel("x / cm")
        ax_2d_x_z.set_ylabel("y / cm")
        ax_2d_y_z.set_xlabel("y / cm")
        ax_2d_y_z.set_ylabel("z / cm")

        ax_3d.set_xlabel("x / cm")
        ax_3d.set_ylabel("y / cm")
        ax_3d.set_zlabel("z / cm")
        if not os.path.exists("./fwf-3d-plot-movie"):
            os.mkdir("./fwf-3d-plot-movie")

        figs_2d.suptitle("Test: Planar Plots of 3D Head.")

        figs_2d.savefig("test-planar-plots.png", dpi=300)
        ax_3d.view_init(azim=270)
        ax_3d.set_title("Test: 3D Head Plots.")
        fig_3d.savefig("test-3d-plot-back-of-head.png", dpi=400)
        ax_3d.plot_surface(sphere_x, sphere_y, sphere_z, color="yellow", zorder=10)
        ax_3d.view_init(azim=90)
        fig_3d.savefig("test-3d-plot-front-of-head.png", dpi=400)
