In the creo_pool folder there are datasets for grabbing rods in a reserach pool. The datasets are split by datasets in a calm pool, or a highly turbulent one (called "under currents").

This repository is for a small study about identifying and rejecting the disturbances in the motion of the bluerov, decribed as follows:

Assumption 1: The linear acceleration of the BlueRov A (and measured by the imu) is the sum of the acceleration due to the robots thursters Phi(t), and the periodic distrubances of the environment Delta(t).
Assumtion 2: The axis are decoupled. IE at time t the A_x is not affected by A_y or A_z at any other time. Therefore it is a linear system.
Assumtion 3: A(t) is time variant. This means it is not a linear time invariant system.
Assumption 4: Phi(t) is linear time invariant, and in face is better represented as Phi(u(t)), where u is the control signal to the thursters at time t. Since the thrusters are fixed, there should be some constant matrix that represents Phi(u(t)).
Assumption 5: Delta(t) can be represented as a sum of sines and cosines, IE taking the fourier transform of Delta(t) should yield some set of coefficients C(t) that represent the amplitude of the different frequency components. Formally C(t) = F{Delta(t)}.
Assumption 6: C(t) is a set of real numbers only, no imaginary component in the pertubations.

From the above it stands that Delta(t) is A(t) - Phi(u(t)), and thus C(t)=F{A(t) - Phi(u_t)}.

Of course estimating C(t) at a specific timestep would be difficult. So we propose an unscented kalman filter to better approximate the set C. For practical considerations we also only evaluate the lower frequency components of C, and should ablate 10^-3Hz up to 10Hz.


For claude: Lets first analyze the dataset where there is minimal pertubation in the water (only the pertubatoins due to the bluerov moving in the small pool). Use the data from that and a similar approach to try to see if we can isolate the pertubations in the under current dataset. If our method works, Phi(u(t)) should look similar in both halves of the dataset. I would like an extensive report with ablations, graphs, and other visualizations.
