# Markowitz Portfolio - The mathematical model

## Parameters
| set           	    | description                                                   |
| ----------------------| ------------------------------------------------------------  |
| $A = \{1,...,n\}, \quad n \in \mathbb{N}$    | contains all assets                    |

| constants           	| description                                                   |
| ----------------------| ------------------------------------------------------------  |
| $ret_{i} \in \mathbb{R}$| return of asset $i \in A$                                   |
| $uplim_{i} \in (0,1)$     | upper limit for the weight of  asset $i \in A$            |
| $lowlim_{i} \in (0,1)$     | lower limit for the weight of  asset $i \in A$           |
| $covar_{ij} \in \mathbb{R}$     | covariance of assets $i,j \in A$                         |
| $\Sigma=(covar_{ij})_{ij}$  | covariance matrix                                        |
| $maxvola \in \mathbb{R}$     | maximum tolerated volatility of the portfolio                 |
| $minret \in \mathbb{R}$     | minimum tolerated return of the portfolio                 |
| $d \in \mathbb{N}$     | factor for variable discretisation, variables get split up in $2^{d}$ intervals                |

| variables           	| description                                                   |
| ----------------------| ------------------------------------------------------------  |
| $w_{i} \in (0, 1)$    | weight of asset i in the portfolio                            |


## Model formulations

### for MIP-Solver

#### maximize_return for MIP
The invesment return of the portfolio should be maximised while not violating the maximum volatility. 
Also the asset weights have to sum up to 1. 
$$ max_{w_{i}} \quad \sum_{i=1}^{n} w_{i} \cdot r_{i} $$
$$ s.t. \quad lowlim_{i} \leq w_{i} \leq uplim_{i},  \qquad \forall i \in A $$
$$ \sum_{i=1}^{n} w_{i} = 1 $$
$$ w \Sigma w \leq maxvola  $$
$$ w_{i} \in (0, 1) $$

#### minimize_volatility for MIP
The invesment volatility of the portfolio should be minimised while not violating the minimum return. 
Also the asset weights have to sum up to 1. 
$$ min_{w_{i}} \quad w \Sigma w $$
$$ s.t. \quad lowlim_{i} \leq w_{i} \leq uplim_{i},  \qquad \forall i \in A $$
$$ \sum_{i=1}^{n} w_{i} \cdot r_{i} \geq minret  $$
$$ \sum_{i=1}^{n} w_{i} = 1 $$
$$ w_{i} \in (0, 1) $$


### for QuantumComputation
#### Variable discretisation
Quantum computers can only compute with binary variables. 
In order to still represent the MarkowitzPortfolio problem, we discretize the weight variables. 
So we now choose a constant $d$ for the discretisation. 
We now split the weight variable for asset i into $2^{d}$ variables.

E.g. we have asset 1 with expected return 10% that has lower-limit  $lowlim_{1} = 0.05$ and upper-limit $uplim_{1} = 0.46$ on its weight and we choose $d=3$ for the discretisation.
Also to avoid the additional constraints with the upperlimit we can integrate them straight into the discretisation.
We replace variable $w_{1}$ in the model with the following: 
$$ w_{1} \in (0.05, 0.46) \quad \Longrightarrow \quad 0.05+(0.46-0.05) \cdot ( \frac{1}{2} w_{1,1} + \frac{1}{4} w_{1,2} + \frac{1}{8} w_{1,3} + \frac{1}{8} w_{1,4} ) $$

#### maximize_return for QC
The invesment return of the portfolio should be maximised while not violating the maximum volatility. 
Also the asset weights have to sum up to 1. 
$$ max_{w_{i_j}} \quad \sum_{i=1}^{n} w_{i} \cdot r_{i} $$
$$ s.t. \quad w \Sigma w \leq maxvola  $$
$$ \sum_{i=1}^{n} w_{i} = 1 $$
$$ lowlim_{i} + (uplim_{i}-lowlim_{i}) \cdot (\frac{1}{2^{d}} w_{i,d+1} + \sum_{j=1}^{d} \frac{1}{2^{j}} w_{i,j}) = w_{i} \qquad \forall i \in A $$
$$ w_{i,j} \in \{0, 1\} \quad \forall i \in A, j \in \{1,...,d+1\} $$

When you reformulate this IP to a QUBO the volatility constraint would unfortunately not be quadratic.
To tackle that issue, we also can incorporate the volatility completely as a penalty in the objective function with a penalty factor:
$$ max_{w_{i,j}} \quad \sum_{i=1}^{n} w_{i} \cdot r_{i}  + penalty \cdot w \Sigma w $$
$$ s.t. \sum_{i=1}^{n} w_{i} = 1 $$
$$ lowlim_{i} + (uplim_{i}-lowlim_{i}) \cdot (\frac{1}{2^{d}} w_{i,d+1} + \sum_{j=1}^{d} \frac{1}{2^{j}} w_{i,j}) = w_{i} \qquad \forall i \in A $$
$$ w_{i,j} \in \{0, 1\} \quad \forall i \in A, j \in \{1,...,d+1\} $$

To completely avoid that issue we can also minimize volatility while considering the return as a linear constraint.

#### minimize_volatility for QC
The invesment volatility of the portfolio should be minimised while not violating the minimum return. 
Also the asset weights have to sum up to 1. 
$$ min_{w_{i,j}} \quad w \Sigma w  $$
$$ s.t. \sum_{i=1}^{n} w_{i} \cdot r_{i} \geq minret  $$
$$ \sum_{i=1}^{n} w_{i} = 1 $$
$$ lowlim_{i} + (uplim_{i}-lowlim_{i}) \cdot (\frac{1}{2^{d}} w_{i,d+1} + \sum_{j=1}^{d} \frac{1}{2^{j}} w_{i,j}) = w_{i} \qquad \forall i \in A $$
$$ w_{i,j} \in \{0, 1\} \quad \forall i \in A, j \in \{1,...,d+1\} $$

When you just relax the inequality-min-return-constraint to an equality-constraint, this model only has 2 linear equality constraints thus is perfect for QUBO reformulation.



  
