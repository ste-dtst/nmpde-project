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
 */



// This project is a modification of Step 26 of the deal.II tutorial.
// Instead of using Rothe's method, we will use the method of lines
// to solve the heat equation. The boundary conditions will be imposed
// via Nitsche's method. For the moment, no grid refinement will
// be performed. For the integration in time, we will rely on the ARKode
// package that is part of the SUNDIALS suite.

// We include some standard deal.II headers
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

// We include some headers to deal with parameters and
// functions which are read from an input file
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>

// This is needed to catch the exception when the parameters file does not exist
#include <deal.II/base/path_search.h>

// This will be needed to integrate in time
#include <deal.II/sundials/arkode.h>

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
        prm.add_parameter("Gamma for Nitsche's method", gamma);
      }
      prm.leave_subsection();

      prm.enter_subsection("ARKode solver parameters");
      {
        prm.add_parameter("Initial time", initial_time);
        prm.add_parameter("Final time", final_time);
        prm.add_parameter("Initial step size", initial_step_size);
        prm.add_parameter("Number of solution outputs", nsteps);
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
      // If .prm file does not exist, it is created with default values and then
      // parsed
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
    double refinement_bottom_fraction = 0.0;

    // This is the gamma parameter for the penalty term in Nitsche's method
    double gamma = 20.0;

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
    unsigned int nsteps = 100;

    // Function objects for the functions involved in the problem.
    // Since we will need to evaluate them at different times,
    // we declare the time-dependent ones mutable, in order to
    // use a const HeatParameters object in the main program.
    mutable FunctionParser<dim> exact_solution;
    FunctionParser<dim>         initial_value;
    mutable FunctionParser<dim> boundary_value;
    mutable FunctionParser<dim> right_hand_side;

    // Additional flag to check if the exact solution is known.
    // If it is, we will compute the error at final_time.
    bool sol_is_known;

    // mutable ParsedConvergenceTable convergence_table;

    ParameterHandler prm;
  };



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
    solve_ode();
    void
    output_results(const Vector<double> &sol, const unsigned int step_no) const;
    // void refine_mesh(const unsigned int min_grid_level,
    //                  const unsigned int max_grid_level);

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

    // Vector<float>  estimated_error_per_cell;
    // Vector<float>  kelly_estimated_error_per_cell;
  };



  template <int dim>
  HeatEquation<dim>::HeatEquation(const HeatParameters<dim> &par)
    : par(par)
    , fe(par.fe_degree)
    , dof_handler(triangulation)
  {}



  template <int dim>
  void
  HeatEquation<dim>::setup_ode()
  {
    // As usual, we distribute the degrees of freedom
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    // Useless call at the moment (there's no refinement)
    // constraints.clear();
    // DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    // constraints.close();

    // We allocate the sparsity pattern for the matrices
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    // DoFTools::make_sparsity_pattern(dof_handler,
    //                                 dsp,
    //                                 constraints,
    //                                 /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    // Finally, we initialize the matrices and the solution vector
    mass_matrix.reinit(sparsity_pattern);
    jacobian_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
  }



  template <int dim>
  void
  HeatEquation<dim>::assemble_ode_matrices()
  {
    // We assemble the mass matrix and the matrix for the implicit part
    // of the ODE in the usual way.

    // Declare the quadrature formulas
    FEValues<dim> fe_values(fe,
                            QGauss<dim>(fe.degree + 1),
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    // Set the usual cell-related objects
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_jacobian_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        // Initialize the fe_values to current cell
        fe_values.reinit(cell);

        // Initialize the cell matrices to zero
        cell_mass_matrix     = 0;
        cell_jacobian_matrix = 0;

        // The usual loop over quadrature points
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              {
                cell_mass_matrix(i, j) +=
                  (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                   fe_values.shape_value(j, q_index) * // phi_j(x_q)
                   fe_values.JxW(q_index));            // dx

                cell_jacobian_matrix(i, j) -=
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx
              }

        // Loop over faces at the boundary of the current cell (if there are
        // any)
        for (auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              // Initialize the fe_face_values to current face
              fe_face_values.reinit(cell, face);

              // Loop over face quadrature points. The formulas are a bit
              // condensed, in order to do less arithmetic operations.
              for (const unsigned int q_index :
                   fe_face_values.quadrature_point_indices())
                for (const unsigned int i : fe_face_values.dof_indices())
                  for (const unsigned int j : fe_face_values.dof_indices())
                    cell_jacobian_matrix(i, j) +=
                      (((fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                           fe_face_values.shape_grad(
                             j, q_index) + // grad phi_j(x_q)
                         fe_face_values.shape_grad(i,
                                                   q_index) * // grad phi_i(x_q)
                           fe_face_values.shape_value(j,
                                                      q_index)) *  // phi_j(x_q)
                          fe_face_values.normal_vector(q_index) -  // n
                        par.gamma / (face->diameter()) *           // gamma/h
                          fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                          fe_face_values.shape_value(j,
                                                     q_index)) * // phi_j(x_q)
                       fe_face_values.JxW(q_index));             // dx
            }

        // Initialize the local dof indices to current cell
        cell->get_dof_indices(local_dof_indices);

        // Distribute local to global
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            {
              mass_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_mass_matrix(i, j));
              jacobian_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_jacobian_matrix(i, j));
            }
      }
  }



  template <int dim>
  void
  HeatEquation<dim>::assemble_ode_explicit_part(const double t)
  {
    // We assemble also the explicit part of the ODE in the usual way.

    // First, we reinit the global vector to zero
    explicit_part.reinit(dof_handler.n_dofs());

    // Declare the quadrature formulas
    FEValues<dim> fe_values(fe,
                            QGauss<dim>(fe.degree + 1),
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    // Set the cell-related objects
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double> cell_explicit_part(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Set the right-hand side and boundary value's time to t
    par.right_hand_side.set_time(t);
    par.boundary_value.set_time(t);

    // Loop over cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        // Initialize the fe_values to current cell
        fe_values.reinit(cell);

        // Initialize the cell vector to zero
        cell_explicit_part = 0;

        // The usual loop over quadrature points
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            const auto &x_q = fe_values.quadrature_point(q_index);
            for (const unsigned int i : fe_values.dof_indices())
              cell_explicit_part(i) +=
                (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                 par.right_hand_side.value(x_q) *    // f(x_q)
                 fe_values.JxW(q_index));            // dx
          }

        // Loop over faces at the boundary of the current cell (if there are
        // any)
        for (auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              // Initialize the fe_face_values to current face
              fe_face_values.reinit(cell, face);

              // Loop over face quadrature points. The formula is a bit
              // condensed, in order to do less arithmetic operations.
              for (const unsigned int q_index :
                   fe_face_values.quadrature_point_indices())
                {
                  const auto &x_q = fe_face_values.quadrature_point(q_index);
                  for (const unsigned int i : fe_face_values.dof_indices())
                    cell_explicit_part(i) +=
                      ((par.gamma / (face->diameter()) *           // gamma/h
                          fe_face_values.shape_value(i, q_index) - // phi_i(x_q)
                        fe_face_values.shape_grad(i,
                                                  q_index) * // grad phi_i(x_q)
                          fe_face_values.normal_vector(q_index)) * // n
                       par.boundary_value.value(x_q) *             // g(x_q)
                       fe_face_values.JxW(q_index));               // dx
                }
            }

        // Initialize the local dof indices to current cell
        cell->get_dof_indices(local_dof_indices);

        // Distribute local to global
        for (const unsigned int i : fe_values.dof_indices())
          explicit_part(local_dof_indices[i]) += cell_explicit_part(i);
      }
  }



  template <int dim>
  void
  HeatEquation<dim>::solve_ode()
  {
    // We set some additional data for the solver. In particular,
    // we want to exploit the fact that the mass matrix and the
    // Jacobian matrix are independent of time.
    const double out_prd = (par.final_time - par.initial_time) / par.nsteps;
    SUNDIALS::ARKode<Vector<double>>::AdditionalData data(
      par.initial_time,        // Initial time
      par.final_time,          // Final time
      par.initial_step_size,   // Initial step size
      out_prd,                 // Output period
      par.minimum_step_size,   // Minimum step size
      par.maximum_order,       // Maximum order
      10,                      // Maximum nonlinear iterations (irrelevant)
      true,                    // implicit_function_is_linear
      true,                    // implicit_function_is_time_independent
      true,                    // mass_is_time_independent
      3,                       // anderson_acceleration_subspace (irrelevant)
      par.absolute_tolerance,  // Absolute tolerance
      par.relative_tolerance); // Relative tolerance

    // Here we declare the ARKode object
    SUNDIALS::ARKode<Vector<double>> ode(data);

    // We have to tell the solver how to multiply the mass matrix
    // by a vector v
    ode.mass_times_vector =
      [&](const double t, const Vector<double> &v, Vector<double> &Mv) {
        (void)t;
        mass_matrix.vmult(Mv, v);
      };

    // We set the explicit function
    ode.explicit_function =
      [&](const double t, const Vector<double> &y, Vector<double> &explicit_f) {
        (void)y;
        assemble_ode_explicit_part(t);
        explicit_f = explicit_part;
      };

    // We set the implicit function
    ode.implicit_function =
      [&](const double t, const Vector<double> &y, Vector<double> &implicit_f) {
        (void)t;
        jacobian_matrix.vmult(implicit_f, y);
      };

    // Since the implicit function is linear, we can supply
    // the Jacobian matrix directly to solve the implicit part
    // of the ODE with a single Newton iteration.
    ode.jacobian_times_vector = [&](const Vector<double> &v,
                                    Vector<double>       &Jv,
                                    const double          t,
                                    const Vector<double> &y,
                                    const Vector<double> &fy) {
      (void)t;
      (void)y;
      (void)fy;
      jacobian_matrix.vmult(Jv, v);
    };

    // When mass_matrix is not the identity matrix, we must
    // provide a linear solver for the system Mx = b
    // through ode.solve_mass.
    // The default linear solver for SUNDIALS is a preconditioned
    // version of GMRES. However, the mass matrix is symmetric and
    // positive definite, therefore we can use a preconditioned
    // conjugate gradient method. We have to specify this to the solver.
    ode.solve_mass = [&](SUNDIALS::SundialsOperator<Vector<double>>       &op,
                         SUNDIALS::SundialsPreconditioner<Vector<double>> &prec,
                         Vector<double>                                   &x,
                         const Vector<double>                             &b,
                         double tol) {
      // We forget about op and prec, since we want to provide
      // our custom preconditioner and solver
      (void)op;
      (void)prec;

      // We provide the solver and the preconditioner in the
      // same way as we have done in previous tutorial programs
      SolverControl            solver_control(1000, tol);
      SolverCG<Vector<double>> solver(solver_control);

      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(mass_matrix, 1.2);

      solver.solve(mass_matrix, x, b, preconditioner);
    };

    // Also the matrix of the linearized system is
    // positive definite for any legal choice of gamma.
    // Therefore, we should use CG also for the implicit part of the ODE.
    // However, I've encountered some problems in supplying the
    // ode.solve_linearized_system function, see temp_dummy.cc
    // in the directory other_files.

    // We supply a function for the output at fixed time intervals
    ode.output_step = [&](const double          t,
                          const Vector<double> &sol,
                          const unsigned int    step_number) {
      std::cout << "Time step " << step_number << " at t = " << t << "."
                << std::endl;
      std::cout << "L_inf norm of solution: " << solution.linfty_norm()
                << std::endl;

      // If the exact solution is known, we compute the error at time t
      if (par.sol_is_known)
        {
          par.exact_solution.set_time(t);
          Vector<double> error(solution.size());
          VectorTools::integrate_difference(dof_handler,
                                            solution,
                                            par.exact_solution,
                                            error,
                                            QGauss<dim>(fe.degree + 1),
                                            VectorTools::L2_norm);
          std::cout << "L2 error: " << error.l2_norm() << std::endl;
        }
      output_results(sol, step_number);
    };

    // Finally, we are ready to solve the ODE
    ode.solve_ode(solution);
  }



  template <int dim>
  void
  HeatEquation<dim>::output_results(const Vector<double> &sol,
                                    const unsigned int    step_no) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(sol, "U");

    data_out.build_patches();

    const std::string filename =
      "output/solution-" + Utilities::int_to_string(step_no, 3) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({step_no, filename});

    std::ofstream pvd_output("output/solution.pvd");

    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }



  template <int dim>
  void
  HeatEquation<dim>::run()
  {
    // Mesh generation, DoF distribution and sparsity pattern allocation
    GridGenerator::hyper_L(triangulation);
    triangulation.refine_global(par.initial_refinement);

    setup_ode();

    // Assembly of the mass matrix and the implicit part of the ODE
    assemble_ode_matrices();

    // We set the initial condition
    VectorTools::interpolate(dof_handler, par.initial_value, solution);

    // ODE solution
    solve_ode();

    // // If the exact solution is known, we compute the error
    // // at par.final_time
    // if (par.sol_is_known)
    //   {
    //     par.exact_solution.set_time(par.final_time);
    //     Vector<double> error(solution.size());
    //     VectorTools::integrate_difference(dof_handler,
    //                                       solution,
    //                                       par.exact_solution,
    //                                       error,
    //                                       QGauss<dim>(fe.degree + 1),
    //                                       VectorTools::L2_norm);
    //     std::cout << "===========================================" <<
    //     std::endl
    //               << "L2 error at final time: " << error.l2_norm() <<
    //               std::endl;
    //   }
  }
} // namespace nmpdeProject


int
main()
{
  try
    {
      using namespace nmpdeProject;

      HeatParameters<2> par;
      HeatEquation<2>   heat_equation_solver(par);
      heat_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
