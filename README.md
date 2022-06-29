# Motion-based Single Anchor Localization

This repository contains the simulations performed for the experiments in the paper [Motion-based Single Anchor Localization](http://resolver.tudelft.nl/uuid:2501fe4c-afc8-4ab4-8085-eb7c077b21f2),
which was written as part of the [Research Project 2022](https://github.com/TU-Delft-CSE/Research-Project) at [TU Delft](https://github.com/TU-Delft-CSE).

The paper introduces a novel motion-based single-anchor localization algorithm and the simulations explore the performance of the proposed algorithm. The algorithm is simulated on a two-robot system, where one of the robots acts as the anchor and the other acts as the target.
For an extensive explanation of the proposed algorithm and problem setup, please see the paper linked above.

## Running the simulations
The simulations are written in Python 3.9 and the requirements can be installed with `pip install -r requirements.txt`

The simulation considered in the paper can be run by executing the following command in the root directory of the repository:

```bash
$ python simulations/circular_path.py
```

This repository also contains a simulation of a random path for the target robot. There are still a few problems with 
running this simulation, but it is still useful for testing the algorithm. The main issue is the fact that the
estimation from the algorithm gets noisier when the target robot moves further away from the anchor, even if the
noise in the measurements is constant. This could be improved
in the future by adapting the Kalman filter. The random path simulation can be run by executing the following command:

```bash
$ python simulations/random_path.py
```