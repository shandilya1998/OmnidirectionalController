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
- `create_training_data_v2(**kwargs)` in [trainingutils](utils/data_generator.py) (create next version instead)
- `Quadruped._get_obs(**kwargs)` in [quadsim](simulations/quadruped.py) to be modified to modify observations
- `QuadrupedV3._get_obs(**kwargs)` in [quadsim](simulations/quadruped_v3.py) to be modified to modify observations

`create_training_data_v2(**kwargs)` takes the following parameters:
- `memory_size`: 10 
- `data_gen_granularity`: 1000
- `window_size`: 150
- `input_size_low_level_controller`: 132
- `cpg_param_size`: 16

`create_training_data_v3(**kwargs)` take the following parameters:
- `memory_size`: 10
- `data_gen_granularity`: 1000
- `window_size`: 150
- `input_size_low_level_controller`: 252
- `cpg_param_size`: 16

The following code can be used to create training data.
```
from utils.data_generator import create_training_data_v3
logidr = 'assets/out/results_v4'
datapath = 'assets/out/results_v3'
create_training_data_v3(logdir, datapath)
```

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
### Results V3
#### Experiment 1
- Feedforward fully connected network with the following number of units in each layer was used- `256, 1024, 1024, 512, 32, 16`.
- Parameterized ReLU used as the activation function.
- Batch Size: 500
- Maximum Epoch Size: 100
- Exponentially decreasing Learning Rate starting at 0.0001 decreasing by a factor of `gamma` = 0.9 every `scheduler_update_freq` = 5 epochs
#### Experiment 2
- Feedforward fully connected network with the following number of units in each layer was used- `256, 1024, 1024, 512, 32, 16`.
- Parameterized ReLU used as the activation function.
- Batch Size: 1000
- Maximum Epoch Size: 100
- Exponentially decreasing Learning Rate starting at 0.0001 decreasing by a factor of `gamma` = 0.9 every `scheduler_update_freq` = 5 epochs
#### Experiment 3
- Feedforward fully connected network with the following number of units in each layer was used- `256, 1024, 1024, 512, 32, 16`.
- Parameterized ReLU used as the activation function.
- Batch Size: 500
- Maximum Epoch Size: 200
- Exponentially decreasing Learning Rate starting at 0.0001 decreasing by a factor of `gamma` = 0.9 every `scheduler_update_freq` = 5 epochs

