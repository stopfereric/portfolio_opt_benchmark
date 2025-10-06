
# Usage instructions for QuOpt

## Installation

The QuOpt code is written with **Python 3.10** because 3.11 led to dependency-conflicts with some packages.
Create your Python-Environment with the file that is located in **/requirements/requirements.txt**


## Optimization instruction manual
QuOpt is able solve problems implemented in **/src/problems/** with solving methods implemented in **/src/solvers/**. 


### Single optimization run
First, you have to set up your opt-run configuration: 
- specify the opt-run **problem-configuration** in **/config/config_files/problem_config.json**
- specify the opt-run **solver-configuration** in **/config/config_files/solver_config.json**

The allowed configurations of problems and solvers are specified in **/config/config_validity.json**. 
Further specification you can find in the markdown file in **/config/config_validity.md**.
Examplary problems- and solver-config-files are located in the following folders: **/config/config_files/problem_configs/** and **/config/config_files/solver_configs/**. 

You can start an optimization run by running the code in **/src/run_optimization.py**. 
It takes the default configs as input that are located in the following files:
 **/config/config_files/default_problem_config.json-**,
 **/config/config_files/default_solver_config.json**,
 **/config/config_files/default_report_config.json**,

Here is a superficial overview on what the code's workflow is:
- set up report-folder for your opt-run in **/results/xxxxx/** (e.g. xxxxx looks like: 2024_06_27_13_36_48_Result_QuOpt_MarkowitzPortfolio_SimulatedAnnealer_Simulated_Annealer_Dwave)
- **read** the problem_config.json and the solver_config.json
- **create the optimization problem** and its specifics in **/src/problems/xxxxx.py**
- **map the optimization problem** to a fitting optimization model, also in **/src/problems/xxxxx.py**
- **run the solver** on this optimization model in **/src/solvers/xxxxx.py**
- **analyse** the solver result in **/src/solvers/xxxxx.py**

During this, all the important information like time stamps and solution quality is written into a report-object that
was created in **/src/report.py**. 

At the end of the optimization run, you can look into the **/results/xxxxx/** folder and realize that some information 
about the optimization run has been saved here:
- **problem_config.json**-file, **solver_config.json**-file and **report_config.json**-file with the respective configuration for this opt-run
- a calculation log: **QuOpt.log**
- potentially there are **model.lp**-files to check the modelling of the opt-problem (-->MIP- and QUBO-lp-files)
- some **.png**-graphics that visualize the opt-results 
- **report.json**-file that contains all the opt-run time stamps and calculated KPIs


### Multiple optimization runs
If you don't want to execute just one optimization run but multiple ones, 
like for example you want to **benchmark** different solving methods for one problem, 
there exists an option to execute multiple optimization runs.

To do that, you have to specify the problem-solver-report combinations for the wanted opt-runs in the benchmark-config-file: **/config/config_files/benchmark_config.json**. 
For some exemplary benchmark-config-files, you can look into the folder **/config/config_files/benchmark_configs/**. 
The specified problem-, solver- and report-configs in the benchmark_config.json have to exist in the folders **/config/config_files/problem_configs/** and **/config/config_files/solver_configs/**.

Then you can execute the benchmark run by running the code in **/src/run_multiple_benchmarks.py**.

Report-folders for each run will be created just like in the single optimization runs in **/results/xxxxx/**.


### Unittest runs
All the unittesting features are located in the following folder: **/test/**.

By the time of writing this instruction manual, there just exists an **executable-test** that checks if some rudimental problem instances can be solved with all implemented solvers.

To execute the executable unittest, you have to specify the config-file in **/test/unittest_executable_config.json**.
This file should have the **same structure** as the config-files for multiple-optimization-runs like described in the previous section.

Then you can start the unittest by running the code in **/test/Unittest.py**.
This will run through all the specified problem-solver-combinations in the unittest_executable_config.json and execute the opt_runs.

You can check the results of your unittest in the created unittest-result-folder in **/results/xxxxx_unittest_run/**
There you will be able to see the **xxxx_test_result.json**-files with the test-results.

### API Tokens
add your API Tokens to the already existing file **/config/config_files/solver_api_tokens.json**.
They are necessary for the access of Dwave and IBM hardware.
You don't want to share your tokens with others and don't want to push it to the git. 

# License

## Commercial licenses

In our code we used Gurobi with the **gurobipy**-library.
For that to work, you have to make sure to install Gurobi separately under its license.
However, there are options for free trials.
For more information: https://www.gurobi.com/faqs/gurobipy/

## OSS libraries


| package         | version    | license |
----------------- | -------------- | ----------------- |
matplotlib  |  3.10.6  |  PSF-2.0 |
typing-extensions  |  4.15.0  |  PSF-2.0 |
certifi  |  2025.8.3  |  MPL-2.0 |
pillow  |  11.3.0  |  MIT-CMU |
cffi  |  2.0.0  |  MIT-0 |
mqt-core  |  3.2.1  |  MIT |
mqt-ddsim  |  2.0.0  |  MIT |
pyparsing  |  3.2.5  |  MIT |
pyspnego  |  0.12.0  |  MIT |
typing-inspection  |  0.4.2  |  MIT |
annotated-types  |  0.7.0  |  MIT |
beautifulsoup4  |  4.14.2  |  MIT |
charset-normalizer  |  3.4.3  |  MIT |
curl-cffi  |  0.13.0  |  MIT |
et-xmlfile  |  2.0.0  |  MIT |
http-sf  |  1.0.4  |  MIT |
openpyxl  |  3.1.5  |  MIT |
peewee  |  3.18.2  |  MIT |
platformdirs  |  4.4.0  |  MIT |
plucky  |  0.4.3  |  MIT |
pydantic  |  2.11.9  |  MIT |
pydantic-core  |  2.33.2  |  MIT |
pyjwt  |  2.10.1  |  MIT |
pyscipopt  |  5.6.0  |  MIT |
pytz  |  2025.2  |  MIT |
setuptools  |  80.9.0  |  MIT |
six  |  1.17.0  |  MIT |
soupsieve  |  2.8  |  MIT |
urllib3  |  2.5.0  |  MIT |
zipp  |  3.23.0  |  MIT |
fonttools  |  4.60.1  |  MIT |
frozendict  |  2.4.6  |  LGPL-3.0; LGPL-3.0-only |
requests-ntlm  |  1.3.0  |  ISC |
markupsafe  |  3.0.3  |  BSD-3-Clause |
antlr4-python3-runtime  |  4.13.2  |  BSD-3-Clause |
authlib  |  1.6.4  |  BSD-3-Clause |
click  |  8.3.0  |  BSD-3-Clause |
contourpy  |  1.3.3  |  BSD-3-Clause |
cycler  |  0.12.1  |  BSD-3-Clause |
dill  |  0.4.0  |  BSD-3-Clause |
idna  |  3.10  |  BSD-3-Clause |
kiwisolver  |  1.4.9  |  BSD-3-Clause |
networkx  |  3.5  |  BSD-3-Clause |
numpy  |  2.3.3  |  BSD-3-Clause |
protobuf  |  6.32.1  |  BSD-3-Clause |
psutil  |  7.1.0  |  BSD-3-Clause |
pycparser  |  2.23  |  BSD-3-Clause |
scipy  |  1.16.2  |  BSD-3-Clause |
websockets  |  15.0.1  |  BSD-3-Clause |
werkzeug  |  3.1.3  |  BSD-3-Clause |
pandas  |  2.3.3  |  BSD-3-Clause |
pysocks  |  1.7.1  |  BSD |
multitasking  |  0.0.12  |  Apache-2.0 |
openqasm3  |  1.0.1  |  Apache-2.0 |
yfinance  |  0.2.66  |  Apache-2.0 |
orjson  |  3.11.3  |  Apache-2.0; MIT |
python-dateutil  |  2.9.0  |  Apache-2.0; BSD-3-Clause |
packaging  |  25.0  |  Apache-2.0; BSD-3-Clause |
cryptography  |  46.0.2  |  Apache-2.0 or BSD-3-Clause |
fasteners  |  0.20  |  Apache-2.0 |
minorminer  |  0.2.19  |  Apache-2.0 |
rustworkx  |  0.17.1  |  Apache-2.0 |
ibm-platform-services  |  0.69.0  |  Apache-2.0 |
dimod  |  0.12.21  |  Apache-2.0 |
diskcache  |  5.6.3  |  Apache-2.0 |
docplex  |  2.30.251  |  Apache-2.0 |
dwave-cloud-client  |  0.14.0  |  Apache-2.0 |
dwave-greedy  |  0.3.0  |  Apache-2.0 |
dwave-networkx  |  0.8.18  |  Apache-2.0 |
dwave-optimization  |  0.6.6  |  Apache-2.0 |
dwave-preprocessing  |  0.6.10  |  Apache-2.0 |
dwave-samplers  |  1.6.0  |  Apache-2.0 |
dwave-system  |  1.33.0  |  Apache-2.0 |
dwave-tabu  |  0.5.0  |  Apache-2.0 |
homebase  |  1.0.1  |  Apache-2.0 |
ibm-cloud-sdk-core  |  3.24.2  |  Apache-2.0 |
importlib-metadata  |  8.7.0  |  Apache-2.0 |
qiskit  |  2.2.1  |  Apache-2.0 |
qiskit-aer  |  0.17.2  |  Apache-2.0 |
qiskit-ibm-runtime  |  0.42.0  |  Apache-2.0 |
qiskit-optimization  |  0.7.0  |  Apache-2.0 |
qiskit-qasm3-import  |  0.6.0  |  Apache-2.0 |
requests  |  2.32.5  |  Apache-2.0 |
stevedore  |  5.5.0  |  Apache-2.0 |
tzdata  |  2025.2  |  Apache-2.0 |

