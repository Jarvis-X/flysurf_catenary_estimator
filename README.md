Implementation of the CatenaryNetwork class

For test, in the main function:
- Initialize CatenaryNetwork with 
  - 3D Positions of actuators (quadrotors)
  - Actuator indices to connect with catenary curves
  - Lengths of the curves
  
- Update in each iteration with new positions of the actuators

- Call get_samples to obtain positions on the catenary curves that are equally spaced on each curve

- Replace `np.mean(desired_sample_positions - sample_points)` with AHD or other error metrics
