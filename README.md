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

## Data Visualization
The relationship between the following pairs was examined for each oscillator
- v<sub>x</sub> vs $\mu$
- v<sub>y</sub> vs $\mu$
- $\omega$<sub>Rz</sub> vs $\mu$
- v<sub>x</sub> vs $\omega$
- v<sub>y</sub> vs $\omega$
- $\omega$<sub>Rz</sub> vs $\omega$
- v<sub>x</sub> vs $\z$<sub>R</sub>
- v<sub>y</sub> vs $\z$<sub>R</sub>
- $\omega$<sub>Rz</sub> $\z$<sub>R</sub>
- v<sub>x</sub> vs $\z$<sub>I</sub>
- v<sub>y</sub> vs $\z$<sub>I</sub>
- $\omega$<sub>Rz</sub> vs $\z$<sub>I</sub>
