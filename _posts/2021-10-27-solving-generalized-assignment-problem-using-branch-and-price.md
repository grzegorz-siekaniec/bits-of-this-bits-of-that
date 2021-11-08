---
layout: post
title: Solving Generalized Assignment Problem using Branch-And-Price
date: 2021-10-27 21:52 +0200
---



{% include toc.md %}

# Generalized Assignment Problem

One of the best known and widely researched problems in combinatorial optimization is Knapsack Problem: 
* given a set of items, each with its weight and value, 
* select a set of items maximizing total value
* that can fit into a knapsack while respecting its weight limit. 

The most common variant of a problem, called 0-1 Knapsack Problem can be formulated as follows:

$$
\begin{array}{lrcl}
\max & \displaystyle\sum_{i=1}^{m} v_i x_i &\\
\textrm{subject to} & \displaystyle \sum_{i=1}^{m} w_i x_i & \le & W\\
\end{array}
$$

where
* $$m$$  - number of items;
* $$x_i$$ - binary variable indicating whether item is selected;
* $$v_i$$ - value of each items;
* $$w_i$$ - weight of each items;
* $$W$$ - maximum weight capacity.


As often happens in mathematics, or science in general, an obvious question to ask is how the problem can be generalized. One of generalization is Generalized Assignment Problem. It answers question - how to find a maximum profit assignment of $$m$$ tasks to $$n$$ machines such that each task ($$i=0, \ldots, m$$) is assigned to exactly one machine ($$j=1, \ldots, n$$), and one machine can have multiple tasks assigned to subject to its capacity limitation.

$$
\begin{array}{lrclll}
\max & \displaystyle\sum_{i=0}^{m} \sum_{j=1}^{n} v_{ij} x_{ij} & & &\\
\textrm{subject to} & \displaystyle \sum_{j=1}^{n} x_{ij} & =& 1 & i=1, \ldots, m & \textrm{(task assignment)} \\
 & \displaystyle  \sum_{i=1}^{m} w_{ij} x_{ij} & \le & c_j & j=1, \ldots, n & \textrm{(machine capacity)} \\
\end{array}
$$

where
* $$n$$  - number of machines;
* $$m$$  - number of tasks;
* $$x_{ij}$$ - binary variable indicating whether task $$i$$ is assigned to machine $$j$$;
* $$v_{ij}$$ - value/profit of assigning task $$i$$ to machine $$j$$;
* $$w_{ij}$$ - weight of assigning task $$i$$ to machine $$j$$;;
* $$c_j$$ - capacity of machine $$j$$.

# Branch-and-price

Branch-and-price is generalization of branch-and-bound method to solve integer programs (IPs),mixed integer programs (MIPs) or binary problems. Both branch-and-price, branch-and-bound, and also branch-and-cut, solve LP relaxation of a given IP. The goal of branch-and-price is to tighten LP relaxation by generating a subset of profitable columns associated with variables to join the current basis.

Branch-and-price builds at the top of branch-and-bound framework. It applies column generation priori to branching. Assuming maximization problem, branching occurs when:
 * Column Generation is finished (i.e. no profitable columns can be found).
 * Objective value of the current solution is greater than best lower bound.
 * The current solution does *not* satisfy integrality constraints.
 
However, if first two conditions are met but not the third one, meaning the current solution *satisfies* integrality constraints, then the best solution and lower bound are updated (lower bound is tightened) with respectively the current solution and its objective value.

The crucial element needed to apply branch-and-price successfully is to find branching scheme. It is tailored to specific problem to make sure that it does not destroy problem structure and can be used in pricing subproblem to effectively generate columns respecting branching rules that can enter Restricted Master Problem (RMP).

Below is flow diagram describing branch-and-price method:

![Branch-and-Price flow diagram]({{site.baseurl}}/assets/images/2021/OCT/branch-and-price-flow-chart.png)

## Dantzig-Wolfe decomposition

The successful application B&P depends on *tight/strong* model formulation. Model formulation is considered *tight* if solution of its LP relaxation satisfies (frequently) integrality constraints. One of structured approaches to come up with such a formulation is to use Dantzig-Wolfe Decomposition technique. See [1] and [2] for details. We will see example of it applied to Generalized Assignment Problem (GAP).

A conventional/basic formulation was described above. Now, let's try to reformulate problem. Let

$$
S_j = \{\mathbf{x}: \mathbf{x} \in \{0, 1\}^{m} \wedge \mathbf{x}^T \mathbf{w}_j \le c_j \}
$$

be a set containing all feasible solutions to Knapsack problem for $$j$$-th machine. Clearly, $$S_j$$ contains finite number of points, so $$S_j = \{ \mathbf{z}_j^1, \ldots, \mathbf{z}_j^{K_j} \}$$, where $$\mathbf{z}_j^k \in \{0, 1\}^{m}$$. You can think about $$\mathbf{z}_j^k \in \{0, 1\}^{m}$$ as 0-1 encoding of tasks that form $$k$$-th feasible solution for machine $$j$$. Now, let $$S = \{ \mathbf{z}_1^1, \ldots, \mathbf{z}_1^{K_1}, \ldots, \mathbf{z}_n^1, \ldots, \mathbf{z}_n^{K_n} \}$$ be a set of all feasible solution to GAP. It, potentially, contains a very large number of elements. Then, every point $$x_{ij}$$ can be expressed by the following convex combination:

$$
\begin{array}{rcll}
x_{ij} & = & \displaystyle \sum_{k=1}^{K_j} z_{ij}^k \lambda_j ^k & \\ 
\displaystyle \sum_{k=1}^{K_j} \lambda_j ^k & = & 1, & j=1,\ldots, n \\
\lambda_i ^k & \in & \{0, 1\} & j=1,\ldots, n, \ k = 1, \ldots, K_j
\end{array}
$$

where $$z_{ij}^k \in \{0, 1\} $$, and $$z_{ij}^k = 1$$ iff task $$i$$ is assigned to machine $$j$$ in $$k$$-th feasible solution for the machine.

Now, let's use this representation to reformulate GAP:

$$
\begin{array}{lrclll}
\max &\displaystyle \sum_{j=1}^{n}\sum_{k=0}^{K_j} \left( v_{ij} z_{ij}^k \right) \lambda_j ^k & & &\\
\textrm{subject to} & \displaystyle \sum_{j=1}^{n}\sum_{k=0}^{K_j} z_{ij}^k \lambda_j ^k  & = & 1 & i=1, \ldots, m & \textrm{(task assignment)} \\
& \displaystyle \sum_{k=0}^{K_j} \lambda_j ^k & = & 1 & j=1, \ldots, n & \textrm{(convexity)} \\
\end{array}
$$

Note that we do not need capacity restrictions as they are embedded into definition of feasible solution for machine $$j$$.

Now that we have formulation that is suitable for column generation, let's turn our attention to it.

## Column generation

Column generation is another crucial component of branch-and-price. There are many great resources devoted to column generation so I will mention only core points:

* Column generation is useful when a problem's pool of feasible solutions contains many elements but only small subset will be present in the optimal solution.
* There exists subproblem (called often pricing problem) that can be used to effectively generate columns that should enter RMP.
* Column generation starts with initial feasible solution.
* Pricing subproblem objective function is updated with dual of the current solution.
* Columns with positive reduced cost, in case of maximization problem, enter problem.
* Procedure continues until such columns exist.

Below is flow diagram describing column generation method:

{:refdef: style="text-align: center;"}
![Column generation flow diagram]({{site.baseurl}}/assets/images/2021/OCT/column-generation-flow-diagram.png)
{: refdef}



# Implementation

Let's see how one approach to implement B&P to solve Generalized Assignment Problem might look like. Below is discussion about main concepts and few code excerpts, a repository containing all code can be found on [github](https://github.com/grzegorz-siekaniec/branch-and-price-gurobi).

```python
@dataclass(frozen=True)
class GeneralAssignmentProblem:

    num_tasks: int
    num_machines: int
    weights: np.ndarray  # shape: num_machines x num_tasks
    profits: np.ndarray  # shape: num_machines x num_tasks
    capacity: np.ndarray  # shape: num_machines x num_tasks
```

An example of problem instance taken from [1] is:

```python
num_machines = 2
num_tasks = 3
profits = np.array([
    [10, 7, 5],
    [6, 8, 11]
])

weights = np.array([
    [9, 6, 3],
    [5, 7, 9]
])

capacity = np.array([11, 18])
```

## Standalone model

It is always good idea to have a reference simple(r) implementation that can be used to validate our results using more sophisticated methods. In our case it is the first, conventional, model. Implementation can be found in repo by checking classes `GAPStandaloneModelBuilder` and `GAPStandaloneModel`. Formulation for a problem instance presented above looks as follows:

```
\ Model gap_standalone_model
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  10 task_0_machine_0 + 6 task_0_machine_1 
   + 7 task_1_machine_0 + 8 task_1_machine_1 
   + 5 task_2_machine_0 + 11 task_2_machine_1
Subject To
 task_assignment_0: task_0_machine_0 + task_0_machine_1 = 1
 task_assignment_1: task_1_machine_0 + task_1_machine_1 = 1
 task_assignment_2: task_2_machine_0 + task_2_machine_1 = 1
 machine_capacity_0: 9 task_0_machine_0 + 6 task_1_machine_0 + 3 task_2_machine_0 <= 11
 machine_capacity_1: 5 task_0_machine_1 + 7 task_1_machine_1 + 9 task_2_machine_1 <= 18
Bounds
Binaries
 task_0_machine_0 task_0_machine_1 
 task_1_machine_0 task_1_machine_1
 task_2_machine_0 task_2_machine_1
End
```

Now let's try to examine building blocks of B&P to discus main part at the end, once all the puzzles are there.

## Initial solution

To start column generation process, we need to have an initial solution. One possible way to derive it is to use two-phase Simplex method. In first step, you add slack variables to each constraint and set objective function as their sum. Then you minimize the problem. If your solution has objective value 0, then first of all you have initial solution and you know that your problem is feasible. In case you end up with positive for any of slack variables, it means that you concluded that your problem is infeasible. You can stop here.

I took a different approach and came up with simple heuristic that generate initial solution. I have not analyzed it thoroughly so I am not sure if it is guaranteed to always return feasible solution if one exists. Its idea is quite simple:
1. Solves a sequence of minimum weight matching problems for bipartite graph:
    1. Construct bipartite graph defined as $$G=(V, A)$$, where $$V = T \cup M$$ - $$T$$ is set of tasks and obviously $$M$$ is set of machines. There exists arc $$a = (t, m)$$ if $$w_{tm} \le rc_{tm}$$, where $$rc_{m}$$ is remaining capacity. Initially remaining capacity is equal to capacity of machine and with each iteration, and assignment of task to machine it is being update. If $$\vert A \vert = 0$$, then stop.
    2. Solves a minimum weight matching problem. 
    3. Updates assignments - say that according to solution task $$t_0$$ should be assigned to machine $$m_0$$, then $$\overline{rc}_{m_0} = rc_{m_0} - w_{t_0 m_0}$$.
2. For every unassigned task - $$t_0$$:
    1. Find a machine where task is contributing with the lowest weight - say machine $$m_0 = \arg\min \{ m: w_{t_0 m} \}$$.
    2. Free up remaining capacity so there is enough space for $$t_0$$ on machine $$m_0$$. Any tasks that was de-assigned in a process is added to pool of unassigned tasks.
    3. Repeat until there are no unassigned tasks.

See details on [github](https://github.com/grzegorz-siekaniec/branch-and-price-gurobi/blob/main/branch-and-price/src/branch_and_price/initial_solution_finder.py#L11).

## Branching rule

As we said before the most important piece needed to implement B&P is branching rules that does not destroy structure of subproblem. Let's consider non-integral solution to RMP. Given convexity constraint it means that there exists machine $$j_0$$ and at least two, and for sake of example say exactly two, $$0 < \lambda_{j_0} ^{k_1} < 1$$ and $$0 < \lambda_{j_0} ^{k_2} < 1$$ such that $$ \lambda_{j_0} ^{k_1} + \lambda_{j_0} ^{k_2} = 1 $$. Since with each of them is connected different assignment (set of tasks), then it leads us to a conclusion that there exists task $$i_0$$ such that $$x_{i_0 j_0} < 1$$ expressed in variables from the original formulation. Now, let's us this information to formulate branching rule:

* left child node: a task $$i_0$$ must be assigned to a machine $$j_0$$.
* right child node: a task $$i_0$$ cannot be assigned to a machine $$j_0$$.

It can be represented by:

```python
@dataclasses.dataclass(frozen=True)
class BranchingRule:
    task: int
    machine: int
    assigned: bool
```

Note that we can use it easily to filter out initial columns for each node that do not satisfy those conditions:
- left child node: column representing assignment of tasks, $$T_j$$, to machine $$j$$ is *kept* if: 
    * $$j = j_0$$ and task $$i_0 \in T_{j_0}$$, or
    * $$j \neq j_0$$ and task $$i_0 \notin T_{j}$$.
- right childe node: column representing assignment of tasks, $$T_j$$, to machine $$j$$ is *filtered out* if: 
    * $$j = j_0$$ and task $$i_0 \in T_{j_0}$$.

See on [github](https://github.com/grzegorz-siekaniec/branch-and-price-gurobi/blob/b58ccc227ea86122a7154227f2bf43e7ed193639/branch-and-price/src/branch_and_price/branch_node.py#L289).

Based on the same principle, subproblem's pool of feasible solution are created - i.e. on left child node:
* knapsack subproblem for machine $$j_0$$ -- variable representing task $$i_0$$ is forced to be $$1$$.
* knapsack subproblem for machine $$j \neq j_0$$ -- variable representing task $$i_0$$ is forced to be $$0$$.

Similarly for right childe node. See on [github](https://github.com/grzegorz-siekaniec/branch-and-price-gurobi/blob/b58ccc227ea86122a7154227f2bf43e7ed193639/branch-and-price/src/branch_and_price/subproblem_builder.py#L78).

## Column generation





# References

[1] Der-San Chen, Robert G. Batson, Yu Dang (2010), Applied Integer Programming - Modeling and Solution, Willey \\
[2] Lasdon, Leon S. (2002), Optimization Theory for Large Systems, Mineola, New York: Dover Publications.
[3] Article