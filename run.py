import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    print('Please ensure you have run "python -m pip install -r requirements.txt" '
          'before attempting to execute this program. \n\nThis command must be run from '
          'within the project root directory.')

    """This method uses finite difference relaxation to find the rest
    positions of hairs within this simulation object.

    Relaxation was used due to instabilities found when f_x != 0
    when using the shooting method as well as the relative ease of
    implementation when compared to other, more compute efficient
    algorithms like collocation.
    """

    """Please note that certain class methods are for internal use only, all those 
    with names preceded with __ are only to be called by other class methods.
    
    The only two class methods with for user consumption are gen_hair_root and 
    simulate_hairs, these do all the legwork necessary to fulfill the coursework.
    """

    """Please also note that this coursework has dependencies outside of those likely 
    installed on the examination machine, please run the printed command in shell/cmd 
    after reading the file: requirements.txt for security purposes.
    
    If this program closes without explanation when running through double-clicks and 
    the like, attempt running in shell/cmd or in your IDE of preference.
    
    As requested in the previous coursework's feedback, I have reduced the frequency of 
    inline comments and ensured a similar level of clarity in variable/attribute names.
    
    In the case of this implementation, __head_radius is a fixed attribute of the 
    DefinitelyHumanHead object and can only be retrieved with the get_head_radius method.
    
    This project uses pathos.multiprocessing, if you wish to change the number of cores 
    in use, please consult DefinitelyHumanHead.__init__ docstring
    """
    # input()
    time.sleep(1)

    from beamhead2 import DefinitelyHumanHead

    # START - TASKS 2 AND 3
    if not os.path.exists("./fwf-plots/"):
        os.mkdir("./fwf-plots/")

    with DefinitelyHumanHead() as head_obj_2d:
        for theta in np.linspace(0.0, np.pi, num=20):
            head_obj_2d.gen_hair_root(theta, length=4)
        # I put the plotting in class but you can extract raw values
        # by accessing the self.simulated_hairs attribute, a list that
        # contains lists of the absolute locations of all hairs on the
        # head as of the most recent simulation run.
        head_obj_2d.simulate_hairs(plot_title=u"Task 2: No Wind", n_finite_elements=50)
        # Setting f_x = 0.1 for Task 3 before re-simulating the environment in object.
        head_obj_2d.f_x = 0.1
        head_obj_2d.simulate_hairs(plot_title=u"Task 3: $f_x = 0.1$", n_finite_elements=50)

    # END - TASKS 2 AND 3

    # START - TASK 4

    with DefinitelyHumanHead(use_phi=True, f_x=0.05) as head_obj_3d:
        # Reduce the magnitude norm of correction vector required for convergence
        # declaration, speeds up the hair simulation greatly.
        head_obj_3d.convergence_mag = 1e-4
        # Create the list of all combinations of theta/phi for the head and add
        # said hairs.
        theta, phi = np.meshgrid(np.linspace(0.0, 0.49 * np.pi, num=10),
                                 np.linspace(0.0, np.pi, num=10))
        theta_phi_combinations = np.vstack([theta.ravel(), phi.ravel()]).T
        for hair_root_loc in theta_phi_combinations:
            head_obj_3d.gen_hair_root(*hair_root_loc)
        # Run the hair simulation with 50 elements in each hair.
        head_obj_3d.simulate_hairs(n_finite_elements=50)

        # 2d planar plot generation
        figs_2d, (ax_2d_x_z, ax_2d_y_z) = plt.subplots(1, 2)
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        phi_angle = np.linspace(0.0, 2 * math.pi, 100)
        theta_angle = np.linspace(0.0, math.pi, 25)

        sphere_x = head_obj_3d.get_head_radius() * np.outer(np.cos(phi_angle),
                                                            np.sin(theta_angle))
        sphere_y = head_obj_3d.get_head_radius() * np.outer(np.sin(phi_angle),
                                                            np.sin(theta_angle))
        sphere_z = head_obj_3d.get_head_radius() * np.outer(np.ones(np.size(phi_angle)),
                                                            np.cos(theta_angle))
        ax_3d.plot_surface(sphere_x, sphere_y, sphere_z, color="yellow")
        x_vals = head_obj_3d.get_head_radius() * np.cos(phi_angle)
        z_vals = head_obj_3d.get_head_radius() * np.sin(phi_angle)
        ax_2d_x_z.plot(x_vals, z_vals, color="blue")
        ax_2d_y_z.plot(x_vals, z_vals, color="blue")
        ax_2d_x_z.set_aspect("equal")
        ax_2d_y_z.set_aspect("equal")
        # Add the hairs to the plots, 2d planes and 3d plot
        for hair in head_obj_3d.simulated_hairs:
            ax_2d_x_z.plot(hair[0], hair[2], color="black")
            ax_2d_y_z.plot(hair[1], hair[2], color="black")
            ax_3d.plot(*hair, color="black", zorder=5)

        if not os.path.exists("./fwf-plots/"):
            os.mkdir("./fwf-plots/")

        ax_2d_x_z.set_xlabel("x / cm")
        ax_2d_x_z.set_ylabel("y / cm")
        ax_2d_y_z.set_xlabel("y / cm")
        ax_2d_y_z.set_ylabel("z / cm")

        ax_3d.set_xlabel("x / cm")
        ax_3d.set_ylabel("y / cm")
        ax_3d.set_zlabel("z / cm")

        figs_2d.suptitle("Task 4: Planar Plots of 3D Head.")

        figs_2d.savefig("fwf-plots/task-4-planar-plots.png", dpi=300)
        # Generates two plots, one from the front of the head and one from the back.
        ax_3d.view_init(azim=270)
        ax_3d.set_title("Task 4: 3D Head Plots.")
        fig_3d.savefig("fwf-plots/task-4-3d-plot-back-of-head.png", dpi=400)
        ax_3d.plot_surface(sphere_x, sphere_y, sphere_z, color="yellow", zorder=10)
        ax_3d.view_init(azim=90)
        fig_3d.savefig("fwf-plots/task-4-3d-plot-front-of-head.png", dpi=400)

    # END - TASK 4
        # Old code that was used to find angles where the head looked reasonable
        # Can be uncommented to get a "plotting movie", unfortunately matplotlib 3d
        # uses only layers to discern whether or not plotted elements are visible, so
        # hairs either show through the sphere or do not show when in front.

        """
        if not os.path.exists("./fwf-3d-plot-movie"):
            os.mkdir("./fwf-plots/fwf-3d-plot-movie")

        count = 0
        print("Generating rotated 3d plot movie.")
        azim_angles, azim_angle_step = np.linspace(0, 360, num=360,
                                                   endpoint=False,
                                                   retstep=True)

        for count_loc in range(len(azim_angles)):
            ax_3d.view_init(azim=azim_angle_step*count_loc)
            fig_3d.savefig(f"./fwf-3d-plot-movie/3d-plot-movie-{count_loc}.png")
            print(f"{count_loc+1}/{len(azim_angles)}")
        """
