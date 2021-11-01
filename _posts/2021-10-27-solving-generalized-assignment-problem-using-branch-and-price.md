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
\max & \sum_{i=1}^{m} v_i x_i &\\
\textrm{subject to} & \sum_{i=1}^{m} w_i x_i & \le & W\\
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
\max & \sum_{i=0}^{m} \sum_{j=1}^{n} v_{ij} x_{ij} & & &\\
\textrm{subject to} & \sum_{j=1}^{n} x_{ij} & =& 1 & i=1, \ldots, m & \textrm{(task assignment)} \\
 & \sum_{i=1}^{m} w_{ij} x_{ij} & \le & d_j & j=1, \ldots, n & \textrm{(machine capacity)} \\
\end{array}
$$

where
* $$n$$  - number of machines;
* $$m$$  - number of tasks;
* $$x_{ij}$$ - binary variable indicating whether task $$i$$ is assigned to machine $$j$$;
* $$v_{ij}$$ - value/profit of assigning task $$i$$ to machine $$j$$;
* $$v_{ij}$$ - weight of assigning task $$i$$ to machine $$j$$;;
* $$d_j$$ - capacity of machine $$j$$.

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