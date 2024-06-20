NMPDE Project
=====================================

This project is a modification of Step 26 of the deal.II tutorial.
Instead of using Rothe's method, we will use the method of lines to solve the heat equation. The boundary conditions will be imposed via Nitsche's method. For the moment, no mesh refinement will be performed. For the integration in time, we will rely on the ARKode package that is part of the SUNDIALS suite.

We will consider the heat equation $u_t(x,t) - \Delta u(x,t) = f(x,t)$ in 2D, with $x$ in a `hyper_L` domain and $t \in (0,0.5)$. In particular, $f$ is the same as in Step 26 and so are the initial condition $u_0(x)$ and the boundary condition $g(x,t)$ (either equal to zero).

The weak form we obtain by using Nitsche's method is the following:

$$
(u_t, v) + (\nabla u, \nabla v) - \langle \nabla u \cdot n,v \rangle - \langle u, \nabla v \cdot n \rangle + \gamma \langle u,v \rangle = (f,v) - \langle g, \nabla v \cdot n \rangle + \gamma \langle g,v \rangle
$$

where $(\cdot,\cdot)$ is the inner product in $\Omega$, $\langle \cdot,\cdot \rangle$ the inner product in $\Gamma = \partial\Omega$ and $n$ is the normal to $\Gamma$.

If $u(x,t) = \sum U_i(t) \phi_i(x)$, this leads to solving the following ODE:

$$
M \mathbf{u}' = f_E(t,u) + f_I(t,u)
$$

where $\mathbf{u}_i=U_i(t)$, $f_I(t,u) = J u$ and

$$
M_{ij}=(\phi_i,\phi_j)
$$

$$
f_E(t,u)_i = (f(\cdot,t),\phi_i) + \gamma \langle g(\cdot,t),\phi_i \rangle - \langle g(\cdot,t),\nabla\phi_i \cdot n \rangle
$$

$$
J_{ij} = -(\nabla\phi_i,\nabla\phi_j) + \langle \phi_i,\nabla\phi_j \cdot n \rangle + \langle \nabla\phi_i \cdot n,\phi_j \rangle - \gamma \langle \phi_i,\phi_j \rangle
$$

In particular, it is clear that the matrices $M$ and $J$ are independent of time, therefore they need to be evaluated only one time (and re-evaluated only when the mesh is changed).

My code is yet to be set up in the context of this deal.II template. At the moment, the main program is in the file source/project.cc and can work on itself with the correct CMakeLists.txt file.


**Possible typos found in the documentation**

- In the documentation of Step 26: in the definition of f(x,t) (paragraph *Testcase*, there's a \tau missing in the second case ($0.5 \le t \le 0.7 \tau$ should be $0.5 \tau \le t \le 0.7 \tau$).

- In the documentation of LinearSolveFunction, SundialsOperator and SundialsPreconditioner are mentioned as arguments, then those are referred to be objects of class LinearOperator. Could this be a typo?

- In the documentation of ARKode, the Detailed Description says that the solve_mass function *can* be provided to use a custom solver instead of the default one (SPGMR), but the documentation of solve_mass says that this function is mandatory if the mass matrix is not the identity. In other words, SPGMR is not used automatically if solve_mass is not provided, and indeed not providing this function results in a runtime exception related to arkStep_FullRHS.

- In the documentation of ARKode and in the file arkode.h, the function mass_preconditioner_solve has 5 arguments, but the documentation says that there's also a sixth parameter gamma.


**Possible ToDo list and some questions**

- Use an AffineConstraints object to distibute local to global and implement adaptive mesh refinement via SolutionTransfer and the solver_should_restart function. Is there any *caveat* in using constraints with two different linear systems to be solved?

- The Jacobian matrix is symmetric. Is it also positive definite for suitable values of the gamma parameter in Nitsche's method? If so, or even if it is not, could it be interesting to provide a custom solver (PCG or MINRES) also for the linearized system? However, I encountered some implementation problems due to gamma (the one in the linearized system) being provided by SUNDIALS but unknown to me. See in other_files/temp_dummy.cc for further explanations.

- (?) Implement a function also for the initial condition (eventually use a parameter handler).

- (?) Use meshloop for the assembly loops

- (?) Consider a different time-dependent problem?

- Is there a way to get from SUNDIALS a "history" of the time step size during the integration process?

- Does deal.II provide a simple way to do the assembly of symmetric matrices by only computing the upper triangular part and then symmetrizing the whole matrix?


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
