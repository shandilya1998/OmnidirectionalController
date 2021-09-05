# Command Line Utility Reference

## Data Generation

- All Values Random
```
#!/bin/sh

LOGDIR="assets/out/results_v2"
CLASS="simulations:Quadruped"
VERSION=0

python3 generate_reference.py --log_dir $LOGDIR --env_class $CLASS --version $VERSION
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
```
- Parameters for Comparison
```
#!/bin/sh

LOGDIR="assets/out/results_v2"
CLASS="simulations:Quadruped"
VERSION=1

python3 generate_reference.py --log_dir $LOGDIR --env_class $CLASS --version $VERSION
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
```

---
**NOTE**
Use the following code to combine two results directories
```
datapaths = [
	'assets/out/results',
	'assets/out/results_v2'
]
from utils.os_utils import _concat_results_v2
_concat_results_v2(datapaths, 'assets/out_results_v3')
```
---

[genref](generate_reference.py) generates the locomotion episodes for the following set of gaits, directions and task.
- **Gait** Trot, LS crawl and DS crawl
- **Tasks** Straight, Turn, Rotate
- **Directions** Left, Right, Forward, Backward
**_NOTE:_**  No `Rotate` episodes were generated for `Trot` gait

Data to be used for training needs to be preprocessed into one to one mapping of robot state and oscillator parameters.
The robot state is a concatenation of the following.
- Desired Goal
- Achieved Goal
- Observation
The Goal `G` is a six dimensional vector containing the linear velocity and angular velocity of the robot
Observation may be modified appropriately to fit need. 
The following modifications in code are currently required to modify observation.
- `create_training_data_v2(**kwargs)` in [trainingutils](utils/data_generator.py)
- `Quadruped._get_obs(**kwargs)` in [quadsim](simulations/quadruped.py) to be modified to modify observations
- `QuadrupedV3._get_obs(**kwargs)` in [quadsim](simulations/quadruped_v3.py) to be modified to modify observations

## Data Visualization
The relationship between the following pairs was examined for each oscillator
- v<sub>x</sub> vs &mu;
- v<sub>y</sub> vs &mu;
- <sup>R</sup>&omega;<sub>z</sub> vs &mu;
- v<sub>x</sub> vs &omega;
- v<sub>y</sub> vs &omega;
- <sup>R</sup>&omega;<sub>z</sub> vs &omega;
- v<sub>x</sub> vs z<sub>R</sub>
- v<sub>y</sub> vs z<sub>R</sub>
- <sup>R</sup>&omega;<sub>z</sub> z<sub>R</sub>
- v<sub>x</sub> vs z<sub>I</sub>
- v<sub>y</sub> vs z<sub>I</sub>
- <sup>R</sup>&omega;<sub>z</sub> vs z<sub>I</sub>
- stability vs &mu;
- d1 vs &mu;
- d2 vs &mu;
- d3 vs &mu;
- stability vs &omega;
- d1 vs &omega;
- d2 vs &omega;
- d3 vs &omega;
- stability vs z<sub>R</sub>
- d1 vs z<sub>R</sub>
- d2 vs z<sub>R</sub>
- d3 vs z<sub>R</sub>
- stability vs z<sub>I</sub>
- d1 vs z<sub>I</sub>
- d2 vs z<sub>I</sub>
- d3 vs z<sub>I</sub>

The following code can be used to generate all of the aforementioned plots.
```
from utils.plot_utils import plot_training_data_v3
logdir = 'assets/out/results'
datapth = logdir
plot_training_data_v3(logdir, datapath)
```

## Supervised Learning
- Train
```
#!/bin/sh

EXPERIMENT=1
DATAPATH="assets/out/results_v3"

python3 supervised_llc.py --experiment $EXPERIMENT --datapath $DATAPATH
rm "$LOGDIR.zip"
zip -r "$LOGDIR.zip" "$LOGDIR/"
```
