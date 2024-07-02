/* ---------------------------------------------------------------------
 *
 * This is a project for the final exam of the course
 * Numerical Methods for Partial Differential Equations,
 * held by prof. Luca Heltai in University of Pisa in 2024.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Stefano Mancini (?), 2024
 * Based on the bare-dealii-app template by Luca Heltai, 2020
 */


// This project is based on Step 26 of the deal.II tutorial.
// Instead of using Rothe's method, we will use the method of lines
// to solve the heat equation. The boundary conditions will be imposed
// via Nitsche's method. For the integration in time, we will rely
// on the ARKode package that is part of the SUNDIALS suite.
// Adaptive mesh refinement will be implemented via solution transfer.


#ifndef dealii_project_h
#define dealii_project_h

// We include the deal.II headers that are needed for the
// declaration of HeatParameters<dim> and HeatEquation<dim>

// Some standard headers
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// We include some headers to deal with parameters and
// functions which are read from an input file
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>

// This is needed to catch the exception when the parameters file does not exist
#include <deal.II/base/path_search.h>

// Standard C++ headers
#include <fstream>
#include <iostream>



namespace nmpdeProject
{
  using namespace dealii;

  // We store the parameters for the problem in a new struct,
  // using a ParameterHandler object to read them from an input file.
  template <int dim>
  struct HeatParameters
  {
    HeatParameters()
    {
      // These are the parameters. Check below for more info
      // on their meaning.
      prm.enter_subsection("Heat equation functions");
      {
        prm.add_parameter(
          "Exact solution expression",
          exact_solution_expression,
          "If you don't know the solution, write   unknown   below.");
        prm.add_parameter("Right hand side expression", rhs_expression);
        prm.add_parameter("Initial value expression", initial_expression);
        prm.add_parameter("Boundary value expression", boundary_expression);
      }
      prm.leave_subsection();

      prm.enter_subsection("FEM parameters");
      {
        prm.add_parameter("Finite element degree", fe_degree);
        prm.add_parameter("Initial global refinement", initial_refinement);
        prm.add_parameter("Refinement top fraction", refinement_top_fraction);
        prm.add_parameter("Refinement bottom fraction",
                          refinement_bottom_fraction);
        prm.add_parameter("Minimum refinement level", min_refinement);
        prm.add_parameter("Maximum refinement level", max_refinement);
        prm.add_parameter("Gamma for Nitsche's method", gamma);
      }
      prm.leave_subsection();

      prm.enter_subsection("ARKode solver parameters");
      {
        prm.add_parameter("Initial time", initial_time);
        prm.add_parameter("Final time", final_time);
        prm.add_parameter("Initial step size", initial_step_size);
        prm.add_parameter("Number of solution outputs", out_steps);
        prm.add_parameter("Minimum step size", minimum_step_size);
        prm.add_parameter("Maximum order", maximum_order);
        prm.add_parameter("Absolute tolerance", absolute_tolerance);
        prm.add_parameter("Relative tolerance", relative_tolerance);
      }
      prm.leave_subsection();

      // prm.enter_subsection("Convergence table");
      // convergence_table.add_parameters(prm);
      // prm.leave_subsection();

      // We try to parse the input file
      try
        {
          prm.parse_input("parameters/heat_" + std::to_string(dim) + "d.prm");
        }
      // If .prm file does not exist, it is created
      // with default values and then parsed
      catch (dealii::PathSearch::ExcFileNotFound &exc)
        {
          std::cout << "Parameters file for " << dim << "D case not found."
                    << std::endl
                    << "I've created one for you with default values."
                    << std::endl
                    << std::endl;
          prm.print_parameters("parameters/heat_" + std::to_string(dim) +
                                 "d.prm",
                               ParameterHandler::KeepDeclarationOrder);
          prm.parse_input("parameters/heat_" + std::to_string(dim) + "d.prm");
        }
      // If something else goes wrong with parsing, then the program is
      // terminated

      // Initialization of the FunctionParser objects.
      // Remark: when a function is time-dependent,
      // we include "t" in the list of variables at the last spot and
      // we set to true the fourth argument "time_dependent".

      // Constants
      std::map<std::string, double> constants;
      constants["pi"] = numbers::PI;

      // We initialize exact_solution only if it is known.
      if (exact_solution_expression != "unknown")
        {
          exact_solution.initialize(
            FunctionParser<dim>::default_variable_names() + ",t",
            {exact_solution_expression},
            constants,
            true);
          sol_is_known = true;
        }
      else
        sol_is_known = false;

      right_hand_side.initialize(FunctionParser<dim>::default_variable_names() +
                                   ",t",
                                 {rhs_expression},
                                 constants,
                                 true);
      initial_value.initialize(FunctionParser<dim>::default_variable_names(),
                               {initial_expression},
                               constants);
      boundary_value.initialize(FunctionParser<dim>::default_variable_names() +
                                  ",t",
                                {boundary_expression},
                                constants,
                                true);
    }

    // Default parameters:

    // Expressions for each function involved in the problem
    std::string exact_solution_expression = "0";
    std::string rhs_expression            = "0";
    std::string initial_expression        = "0";
    std::string boundary_expression       = "0";

    // FE degree
    unsigned int fe_degree = 1;

    // Global refinement steps to be performed before integration
    unsigned int initial_refinement = 3;

    // Parameters for the adaptive mesh refinement
    double refinement_top_fraction    = .3;
    double refinement_bottom_fraction = .1;
    double min_refinement             = 1;
    double max_refinement             = 6;

    // This is the gamma parameter for the penalty term in Nitsche's method
    double gamma = 10.0;

    // Parameters for the ARKode solver
    double       initial_time       = 0.0;
    double       final_time         = 1.0;
    double       initial_step_size  = 1e-2;
    double       minimum_step_size  = 1e-6;
    unsigned int maximum_order      = 5;
    double       absolute_tolerance = 1e-6;
    double       relative_tolerance = 1e-5;

    // In this variable we set the number of time steps, other than
    // initial_time, at which we want to produce a solution output
    unsigned int out_steps = 100;

    // Function objects for the functions involved in the problem.
    // Since we will need to evaluate them at different times,
    // we declare the time-dependent ones mutable, in order to
    // use a const HeatParameters object in the main program.
    mutable FunctionParser<dim> exact_solution;
    FunctionParser<dim>         initial_value;
    mutable FunctionParser<dim> boundary_value;
    mutable FunctionParser<dim> right_hand_side;

    // Additional flag to check if the exact solution is known.
    // If it is, we will compute the L2 error at each output time.
    bool sol_is_known;

    // mutable ParsedConvergenceTable convergence_table;

    ParameterHandler prm;
  };



  // This is the class that we use to solve the heat equation
  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation(const HeatParameters<dim> &parameters);

    void
    run();

  private:
    void
    setup_ode();
    void
    assemble_ode_matrices();
    void
    assemble_ode_explicit_part(const double t);
    void
    refine_mesh(Vector<double>    &sol,
                const unsigned int min_grid_level,
                const unsigned int max_grid_level);
    void
    solve_ode();
    void
    output_results(const Vector<double> &sol, const unsigned int step_no) const;

    const HeatParameters<dim> &par;

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    // AffineConstraints<double> constraints;

    // Here we have the matrices involved in the ODE
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> jacobian_matrix;

    // Here are the solution and the vector for the explicit part
    Vector<double> solution;
    Vector<double> explicit_part;
  };

} // namespace nmpdeProject


#endif