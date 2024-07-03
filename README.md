NMPDE Project
=====================================

This project is based on Step 26 of the deal.II tutorial. It has been set up to work with the *bare-dealii-app* template by Luca Heltai (read below for further info).
Instead of using Rothe's method, we will use the method of lines to solve the heat equation. The boundary conditions will be imposed via Nitsche's method. For the integration in time, we will rely on the ARKode package that is part of the SUNDIALS suite. Adaptive mesh refinement will be implemented via solution transfer.

**The problem**

We consider the heat equation $u_t(x,t) - \Delta u(x,t) = f(x,t)$, with initial condition $u_0(x)$ and boundary condition $g(x,t)$. The domain for the space variable $x$ is a `hyper_L` (2D and 3D case) or the interval $[-1,1]$ (1D case), while the time variable $t$ is in a given interval $[t_0,t_1]$. The user can customize $f$, $u_0$, $g$, $t_0$ and $t_1$ via a .prm file. If also the exact solution $u$ is provided, then the program will compute the $L^2$ error for the numerical solution. In particular, if you want to test a *manufactured solution*, you can use the Jupyter Notebook `other_files/manufactured_heat.ipynb` to compute the correct $f$ automatically, given $u$.

The weak form we obtain by using Nitsche's method is the following:

$$
(u_t, v) + (\nabla u, \nabla v) - \langle \nabla u \cdot n,v \rangle - \langle u, \nabla v \cdot n \rangle + \gamma \langle u,v \rangle = (f,v) - \langle g, \nabla v \cdot n \rangle + \gamma \langle g,v \rangle
$$

where $(\cdot,\cdot)$ is the inner product in $\Omega$, $\langle \cdot,\cdot \rangle$ the inner product in $\Gamma = \partial\Omega$ and $n$ is the normal to $\Gamma$.

If $u(x,t) = \sum U_i(t) \phi_i(x)$, this leads to solving the following ODE:

$$
M \mathbf{u}' = f_E(t,\mathbf{u}) + f_I(t,\mathbf{u})
$$

where $\mathbf{u}_i=U_i(t)$, $f_I(t,\mathbf{u}) = J \mathbf{u}$ and

$$
M_{ij}=(\phi_i,\phi_j)
$$

$$
f_E(t,\mathbf{u})_i = (f(\cdot,t),\phi_i) + \gamma \langle g(\cdot,t),\phi_i \rangle - \langle g(\cdot,t),\nabla\phi_i \cdot n \rangle
$$

$$
J_{ij} = -(\nabla\phi_i,\nabla\phi_j) + \langle \phi_i,\nabla\phi_j \cdot n \rangle + \langle \nabla\phi_i \cdot n,\phi_j \rangle - \gamma \langle \phi_i,\phi_j \rangle
$$

In particular, it is clear that the matrices $M$ and $J$ are independent of time, therefore they need to be evaluated only one time (and re-evaluated only when the mesh is changed).

The .prm file can also be used to customize a variety of parameters for the ARKode solver, as well as $\gamma$, the finite element degree and the mesh refinement strategy.

**Before you start**

The program will look for the following directories to write its output and parameters:

- `output_1d`

- `output_2d`

- `output_3d`

- `parameters`

Make sure to create them in advance in the folder where you will put the executables.

Also, in the code (*at the moment, but may not be necessary*) there is a `if constexpr` statement, which requires your compiler to support at least C++17.


**Some tests - 1D case**

Problem 0.1:

|  		  											| Error at final time | # ARKode steps | Last stepsize |
| ------------------------------------------------- |:------------------: | :------------: | :-----------: |
| Default settings    								| 0.0170169 		  | 239 		   | 0.00863937    |
| Setting `initial_refinement` = 5, `gamma` = 100 	| 0.000897367 		  | 4588 		   | 0.00131488    |

Problem 0.2:

|  		  											| Error at final time | # ARKode steps | Last stepsize |
| ------------------------------------------------- |:------------------: | :------------: | :-----------: |
| Default settings    								| 0.170158	 		  | 223 		   | 0.00248579    |
| Setting `initial_refinement` = 5, `gamma` = 100 	| 0.00905921 		  | 4945 		   | 0.00399241    |

Problem 1.1:

|  		  											| Error at final time | # ARKode steps | Last stepsize |
| ------------------------------------------------- |:------------------: | :------------: | :-----------: |
| Default settings    								| 0.123111	 		  | 186 		   | 0.0015718     |
| Setting `initial_refinement` = 5, `gamma` = 100 	| 0.00599535 		  | 3245 		   | 0.00176189    |

Steeper solutions require finer meshes. In particular, I found that tweaking `gamma` when raising the number of refinements is important, otherwise numerical solutions might explode.


**Some tests - 2D case**

Problem 0.1.

Here, modifying the parameters in a way that "would reduce the error" seems to mess things up a lot. Instead, default settings are better. In any case, the numerical solution does not behave as expected: there is a lot of flickering around the boundary (things get worse when playing with the parameters) and a diagonal pattern emerges on the mesh. I've already seen this at laboratory when talking about mixed elements, so my thoughts are: maybe FE_Q is not the right choice for this problem?

|  		  											| Error at final time | # ARKode steps | Last stepsize |
| ------------------------------------------------- |:------------------: | :------------: | :-----------: |
| Default settings    								| 0.0030494 		  | 635 		   | 0.00560729    |
| (e.g.) Setting `fe_degree` = 2, `gamma` = 100 	| 0.00144382 		  | 2126 		   | 0.000838382   |

<video width="844" height="532" controls>
  <source src="0_1pbm_2d.avi" type="video/avi">
</video>


To_be_completed! (For instance, problems 2.2 and 2.3 are instead well behaved)



**ToDo list**

- Implement adaptive mesh refinement.

- Provide a good PCG solver to SUNDIALS and save some computational resources.

- Use meshloop for the assembly loops.


About this template
=====================================

[![Build Status](https://travis-ci.org/luca-heltai/bare-dealii-app.svg)](https://travis-ci.org/luca-heltai/bare-dealii-app)

[![Build Status](https://gitlab.com/luca-heltai/bare-dealii-app/badges/master/pipeline.svg)](https://gitlab.com/luca-heltai/bare-dealii-app/)


A bare deal.II application, with directory structure, a testsuite, and unittest
block based on google tests.

This repository can be used to bootstrap your own deal.II
application. The structure of the directory is the following:

	./source
	./include
	./tests
	./gtests
	./doc

The directories contain a minimal working application (identical to step-6, 
where implementations and declarations have been separated) to solve the
Poisson problem on a square, a test directory that uses deal.II style testing, 
a test directory that uses google tests, and a doc directory, that contains
a `Doxyfile` to use with `doxygen`.

The `CMakeLists.txt` will generate both some executables and two libraries
containing all cc files **except** `source/main.cc`, one for Debug mode and
one for Release mode. This library is linked to the running tests, so that you 
can make tests on your application just as you would do with the deal.II 
library.

Modify the TARGET variable in the CMakeLists.txt to your application
name. Two libraries named ./tests/lib${TARGET}.so and ./tests/lib${TARGET}.g.so
will be generated together with one executable per dimension, per build type,
i.e., a total of six executables, and two libraries.

After you have compiled your application, you can run 

	make test

or
	
	ctest 

to start the testsuite.

Take a look at
https://www.dealii.org/developer/developers/testsuite.html for more
information on how to create tests and add categories of tests, and a look at
https://github.com/google/googletest/blob/master/googletest/docs/primer.md
for a quick setup of unit tests with google test.

Both `.travis.yml` and `.gitlab-ci.yml` files are provided that 
build the application and run the tests in the tests directory using
ctest, in continuous integration, by running under docker with the 
image provided on dockerhub.com: `dealii/dealii:master-focal`.

Moreover, three github actions are provided to check indentation, build
the documentation, and test the library from within github actions.

The documentation is built and deployed at each merge to master. You can 
find the latest documentation here:
https://luca-heltai.github.io/bare-dealii-app/

Licence
=======

See the file ./LICENSE for details
