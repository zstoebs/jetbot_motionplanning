# Autonomous motion planning for NVIDIA JetBot

Zach Stoebner

## Hardware Setup

The JetBot was built following the documentation on the [JetBot homepage](https://jetbot.org/master/index.html). For the parts with multiple options: the IMX219-160 listed as the second option for cameras, the M2 card + antennas listed as the first option for wifi, and the 65mm wheels listed as the second option for wheels were used. The total cost was approximately $300. The hardware setup time was approximately twelve hours spread between two days. A significant portion of the time was spent extracting a screw terminal from the motor board that was placed incorrectly. [1](/assets/fig1.jpg) shows the completed JetBot hardware assembly.

## Software Setup

The OS for the JetBot was flashed onto a 64 GB SD card in a two step process. First, the NVIDIA Jetson Nano OS was flashed to initialize the Jetson and its the drivers. Second, the JetBot OS was flashed over the Jetson OS on the SD card to initialize the JetBot. On first startup, wifi was configured from the command line; on subsequent startups, the JetBot would automatically connect to the network and could be interfaced through JupyterLab on a browser at the JetBot’s IP address. Total software setup time took about 3.5 hours.

## CasADi on JetBot

The JetBot OS (Ubuntu 18.04 LTS - aarch64) is not supported by any current binary distributions of CasADi.8  With much investigative effort, it was possible to build CasADi from source, mostly following the instructions found on the CasADi GitHub wiki. Total build time took about an hour to complete. The command that yielded a successful build on the JetBot, once all prerequisites and source were installed, was:

```
cmake -DWITH_PYTHON=ON -DWITH_PYTHON3 =ON ..
```

A demonstration of motion planning can be found in [the project notebook](https://github.com/zstoebs/jetbot_motionplanning/blob/main/nonvisual_pathplanning.ipynb). 

## References

[Enzenhofer. "Numerical Solution of Mixed Linear Complementarity Problems in Multibody Dynamics with Contact." 2018](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjszOvY_rH0AhU9CjQIHQFBAx0QFnoECAMQAQ&url=https%3A%2F%2Fescholarship.mcgill.ca%2Fdownloads%2F2z10ws808&usg=AOvVaw3x3QL54b3sOErLRdSzTeO5)

[Dirkse, Steven & Ferris, Michael. (1995). The path solver: a nommonotone stabilization scheme for mixed complementarity problems. Optimization Methods & Software - OPTIM METHOD SOFTW. 5. 123-156. 10.1080/10556789508805606.](https://www.researchgate.net/publication/250889964_The_path_solver_a_nommonotone_stabilization_scheme_for_mixed_complementarity_problems)

[1] Pepy, R., Lambert, A., and Mounier, H., “Path planning using a dynamic vehicle model,” in [2006 2nd International Conference on Information Communication Technologies], 1, 781–786 (2006).
[2] Choset, H., La Civita, M., and Park, J., “Path planning between two points for a robot experiencing local- ization error in known and unknown environments,” (11 1999).
[3] Lozano-Perez, T., “A simple motion-planning algorithm for general robot manipulators,” IEEE Journal on Robotics and Automation 3(3), 224–238 (1987).
[4] Yonetani, R., Taniai, T., Barekatain, M., Nishimura, M., and Kanezaki, A., “Path planning using neural a* search,” in [International Conference on Machine Learning], 12029–12039, PMLR (2021).
[5] Lee, L., Parisotto, E., Chaplot, D. S., Xing, E., and Salakhutdinov, R., “Gated path planning networks,” in [International Conference on Machine Learning], 2947–2955, PMLR (2018).
[6] Mansouri, S. S., Kanellakis, C., Fresk, E., Kominiak, D., and Nikolakopoulos, G., “Cooperative coverage path planning for visual inspection,” Control Engineering Practice 74, 118–131 (2018).
[7] Dirkse, S. and Ferris, M., “The path solver: A non-monotone stabilization scheme for mixed complementarity problems,” Optimization Methods and Software 5 (12 1993).
[8] Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., and Diehl, M., “CasADi – A software framework for nonlinear optimization and optimal control,” Mathematical Programming Computation (In Press, 2018).
[9] Araki, M., “Pid control,” CONTROL SYSTEMS, ROBOTICS, AND AUTOMATION 2.
