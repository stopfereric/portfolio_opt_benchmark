
# Usage instructions for QuOpt

## Installation

The QuOpt code is written with **Python 3.10** because 3.11 lead to dependency-conflicts with some packages

### Option 1: Auto-configure conda environment with bash-file 
There exists a **create_conda_environment.sh**-file that configures thw environment automatically.
Enter the following in Anaconda Prompt, where you replace "req_folder" with the **/requirements**  folder address
on your computer:
```bash
         cd req_folder
```
then execute the bash-file:
```bash
         create_conda_environment.sh
```
This should have created the environments: "quopt"
which you can activate by:
```bash
         activate quopt
```
Then you have fulfilled all the prerequisites for QuOpt!


### Option 2: Use the requirements.txt-file
it is located in **/requirements/requirements.txt**


## Optimization instruction manual
QuOpt is able solve problems implemented in **/src/problems/** with solving methods impolemented in **/src/solvers/**. 


### Single optimization run
First, you have to set up your opt-run configuration: 
- specify the opt-run **problem-configuration** in **/config/config_files/problem_config.json**
- specify the opt-run **solver-configuration** in **/config/config_files/solver_config.json**

The allowed configurations of problems and solvers are specified in **/config/config_validity.json**. Further specification you can find in the markdown file in **/config/config_validity.md**.
If a given configuration in problem_config.json and solver_config.json doesn't comply with the allowed configuration, 
the program will throw an error, so check your config-file according to the information the config_validity.json and -.md .
Examplary problems- and solver-config-files are located in the following folders: **/config/config_files/problem_configs/** and **/config/config_files/solver_configs/**. 
You can just copy the data from the .json-files located in these folders to the **/config/config_files/problem_config.json-** and **/config/config_files/solver_config.json**-files and start some test runs.

Once you've finished setting up your config.json-files, you can start your optimization run by running the code in **/src/run_optimization.py**. 
Here is a superficial overview on what the code's workflow is:
- set up report-folder for your opt-run in **/results/xxxxx/** (e.g. xxxxx looks like: 2024_06_27_13_36_48_Result_QuOpt_AssemblyLineBalancing_SimulatedAnnealer_Simulated_Annealer_Dwave)
- **read** the problem_config.json and the solver_config.json
- **create the optimization problem** and its specifics in **/src/problems/xxxxx.py**
- **map the optimization problem** to a fitting optimization model  in **/src/problems/xxxxx.py**
- **run the solver** on this optimization model in **/src/solvers/xxxxx.py**
- **analyse** the solver result in **/src/solvers/xxxxx.py**

During this, all the important information like time stamps and solution quality is written into a report-object that
was created in **/src/report.py**. 

At the end of the optimization run, you can look into the **/results/xxxxx/** folder and realize that some information 
about the optimization run has been saved here:
- **problem_config.json**-file and **solver_config.json**-file with the respective configuration for this opt-run
- a calculation log: **QuOpt.log**
- potentially there are **model.lp**-files to check the modelling of the opt-problem (-->MIP- and QUBO-lp-files)
- some **.png**-graphics that visualize the opt-results 
- **report.json**-file that contains all the opt-run time stamps and calculated KPIs


### Multiple optimization runs
If you don't want to execute just one optimization run but multiple ones, like for example you want to **benchmark** different solving methods for one problem, there exists an option to execute multiple optimization runs **parallely by using all your computer's processors**.

To do that, you have to specify the problem-solver combinations for the wanted opt-runs in the benchmark-config-file: **/config/config_files/benchmark_config.json**. 
For some examplary benchmark-config-files, you can look into the folder **/config/config_files/benchmark_configs/**. 
The specified problems and solving-methods in the benchmark_config.json have to exist in the folders **/config/config_files/problem_configs/** and **/config/config_files/solver_configs/**, otherwise the code will throw an error.

Then you can execute the benchmark run by running the code in **/src/run_multiple_benchmarks.py**.

Report-folders for each run will be created just like in the single optimization runs in **/results/xxxxx/**.


### Unittest runs
All the unittesting features are located in the following folder: **/test/**.

By the time of writing this instruction manual, there just exists an **executable-test** that checks if some rudimental problem instances can be solved with all implemented solvers.

To execute the executable unittest, you have to specify the config-file in **/test/unittest_executable_config.json**.
This file should have the **same structure** as the config-files for multiple-optimization-runs like described in the previous section.

Then you can start the unittest by running the code in **/test/Unittest.py**.
This will run through all the specified problem-solver-combinations in the unittest_executable_config.json and execute the opt_runs.

You can check the results of your unittest in the created unittest-result-folder in **/test/unittest_results/xxxxx/**
There you will be able to see the **xxxx_test_result.json**-files with the test-results.

### API Tokens
add your API Tokens to the already existing file **/config/config_files/solver_api_tokens.json**.
You don't want to share your tokens with others and don't want to push it to the git. 
In order to do that, open git bash inside **/config/config_files/** and write:
```bash
          git update-index --assume-unchanged solver_api_tokens.json
```
When you want to switch branches, this might lead to a problem that is only solvable when you revoke this 'ignoring'-command:
```bash
git update-index --no-assume-unchanged
```

# License

## Used OSS libraries


| package         | version    | license |
----------------- | -------------- | ----------------- |
PySCIPOpt	 | 	5.1.1	 | 	MIT LICENSE	 | 
contourpy	 | 	1.2.1	 | 	BSD LICENSE	 | 
cycler	 | 	0.12.1	 | 	BSD LICENSE	 | 
dill	 | 	0.3.8	 | 	BSD LICENSE	 | 
dimod	 | 	0.12.16	 | 	APACHE SOFTWARE LICENSE	 | 
docplex	 | 	2.27.239	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-cloud-client	 | 	0.12.0	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-greedy	 | 	0.3.0	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-networkx	 | 	0.8.15	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-optimization	 | 	0.1.0	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-preprocessing	 | 	0.6.5	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-samplers	 | 	1.2.0	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-system	 | 	1.25.0	 | 	APACHE SOFTWARE LICENSE	 | 
dwave-tabu	 | 	0.5.0	 | 	APACHE SOFTWARE LICENSE	 | 
et-xmlfile	 | 	1.1.0	 | 	MIT LICENSE	 | 
fonttools	 | 	4.53.0	 | 	MIT LICENSE	 | 
homebase	 | 	1.0.1	 | 	APACHE 2.0	 | 
ibm-platform-services	 | 	0.54.1	 | 	APACHE SOFTWARE LICENSE	 | 
importlib-resources	 | 	6.4.0	 | 	APACHE SOFTWARE LICENSE	 | 
kiwisolver	 | 	1.4.5	 | 	BSD LICENSE	 | 
matplotlib	 | 	3.9.0	 | 	PYTHON SOFTWARE FOUNDATION LICENSE	 | 
minorminer	 | 	0.2.14	 | 	APACHE SOFTWARE LICENSE	 | 
mqt.ddsim	 | 	1.22.0	 | 	MIT LICENSE	 | 
networkx	 | 	3.3	 | 	BSD LICENSE	 | 
numpy	 | 	1.26.4	 | 	BSD LICENSE	 | 
openpyxl	 | 	3.1.4	 | 	MIT LICENSE	 | 
packaging	 | 	24.1	 | 	APACHE SOFTWARE LICENSE;; BSD LICENSE	 | 
pandas	 | 	2.2.0	 | 	BSD LICENSE	 | 
pillow	 | 	10.3.0	 | 	HISTORICAL PERMISSION NOTICE AND DISCLAIMER (HPND)	 | 
psutil	 | 	6.0.0	 | 	BSD LICENSE	 | 
pydantic	 | 	2.7.4	 | 	MIT LICENSE	 | 
pyparsing	 | 	3.1.2	 | 	MIT LICENSE	 | 
python-dateutil	 | 	2.9.0.post0	 | 	BSD LICENSE;; APACHE SOFTWARE LICENSE	 | 
pytz	 | 	2024.1	 | 	MIT LICENSE	 | 
qiskit	 | 	1.1.1	 | 	APACHE SOFTWARE LICENSE	 | 
qiskit-aer	 | 	0.14.2	 | 	APACHE SOFTWARE LICENSE	 | 
qiskit-algorithms	 | 	0.3.0	 | 	APACHE SOFTWARE LICENSE	 | 
qiskit-ibm-runtime	 | 	0.24.1	 | 	APACHE SOFTWARE LICENSE	 | 
qiskit-optimization	 | 	0.6.1	 | 	APACHE SOFTWARE LICENSE	 | 
requests	 | 	2.32.3	 | 	APACHE SOFTWARE LICENSE	 | 
requests-ntlm	 | 	1.3.0	 | 	ISC LICENSE (ISCL)	 | 
rustworkx	 | 	0.14.2	 | 	APACHE SOFTWARE LICENSE	 | 
scipy	 | 	1.13.1	 | 	BSD LICENSE	 | 
setuptools	 | 	65.5.1	 | 	MIT LICENSE	 | 
six	 | 	1.16.0	 | 	MIT LICENSE	 | 
stevedore	 | 	5.2.0	 | 	APACHE SOFTWARE LICENSE	 | 
symengine	 | 	0.11.0	 | 	MIT LICENSE	 | 
sympy	 | 	1.12.1	 | 	BSD LICENSE	 | 
typing_extensions	 | 	4.12.2	 | 	PYTHON SOFTWARE FOUNDATION LICENSE	 | 
tzdata	 | 	2024.1	 | 	APACHE SOFTWARE LICENSE	 | 
urllib3	 | 	2.2.2	 | 	MIT LICENSE	 | 
yfinance |  0.2.43   |  APACHE SOFTWARE LICENSE	 | 
websocket-client	 | 	1.8.0	 | 	APACHE SOFTWARE LICENSE	 | 




