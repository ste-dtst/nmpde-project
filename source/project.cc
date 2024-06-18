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
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

// This will be needed to integrate in time
#include <deal.II/sundials/arkode.h>

// Standard C++ headers
#include <fstream>
#include <iostream>
 

namespace nmpdeProject
{
  using namespace dealii;
 
 
  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void run();
 
  private:
    void setup_ode();
    void assemble_ode_matrices();
    void assemble_ode_explicit_part(const double t);
    void solve_ode();
    void output_results(const Vector<double> &sol,
                        const unsigned int step_no) const;
    // void refine_mesh(const unsigned int min_grid_level,
    //                  const unsigned int max_grid_level);
 
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
 
    // Global refinement steps to be performed before integration
    const unsigned int initial_global_refinement = 3;

    // Some parameters for the ARKode solver
    double       initial_time = 0.0;
    double       final_time = 0.5;
    const double min_step = 1e-6;  // Minimum step size of the solver

    // In this variable we set the number of time steps, other than
    // initial_time, at which we want to produce a solution output
    unsigned int nsteps = 100;
 
    // This is the gamma parameter for the penalty term in Nitsche's method
    const double gamma = 20;
  };
 
 
 
  // The right-hand side and the boundary values of the heat equation
  // are exactly the same as in Step 26.
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>()
      , period(0.2)
    {}
 
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
 
  private:
    const double period;
  };
 
 
 
  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(dim == 2, ExcNotImplemented());
 
    const double time = this->get_time();
    const double point_within_period =
      (time / period - std::floor(time / period));
 
    if ((point_within_period >= 0.0) && (point_within_period <= 0.2))
      {
        if ((p[0] > 0.5) && (p[1] > -0.5))
          return 1;
        else
          return 0;
      }
    else if ((point_within_period >= 0.5) && (point_within_period <= 0.7))
      {
        if ((p[0] > -0.5) && (p[1] > 0.5))
          return 1;
        else
          return 0;
      }
    else
      return 0;
  }
 
 
 
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };
 
 
 
  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }
 
 
 
  template <int dim>
  HeatEquation<dim>::HeatEquation()
    : fe(1)
    , dof_handler(triangulation)
  {}
 
 
 
  template <int dim>
  void HeatEquation<dim>::setup_ode()
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
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp);
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
  void HeatEquation<dim>::assemble_ode_matrices()
  {
    // We assemble the mass matrix and the matrix for the implicit part
    // of the ODE in the usual way.
    
    // Declare the quadrature formulas
    FEValues<dim> fe_values(fe,
                            QGauss<dim>(fe.degree + 1),
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                    QGauss<dim-1>(fe.degree + 1),
                                    update_values | update_gradients |
                                    update_quadrature_points | update_normal_vectors |
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
        cell_mass_matrix = 0;
        cell_jacobian_matrix = 0;
  
        // The usual loop over quadrature points
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
            {
              cell_mass_matrix(i, j) += (fe_values.shape_value(i, q_index) *  // phi_i(x_q)
                                          fe_values.shape_value(j, q_index) * // phi_j(x_q)
                                          fe_values.JxW(q_index));            // dx

              cell_jacobian_matrix(i, j) -= (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                                              fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                                              fe_values.JxW(q_index));           // dx
            }

        // Loop over faces at the boundary of the current cell (if there are any)
        for (auto &face: cell->face_iterators())
          if (face->at_boundary())
          {
            // Initialize the fe_face_values to current face
            fe_face_values.reinit(cell, face);

            // Loop over face quadrature points. The formulas are a bit condensed,
            // in order to do less arithmetic operations.
            for (const unsigned int q_index : fe_face_values.quadrature_point_indices())
              for (const unsigned int i : fe_face_values.dof_indices())
                for (const unsigned int j : fe_face_values.dof_indices())
                  cell_jacobian_matrix(i, j) += (((fe_face_values.shape_value(i, q_index) * // phi_i(x_q)
                                                  fe_face_values.shape_grad(j, q_index) +   // grad phi_j(x_q)
                                                  fe_face_values.shape_grad(i, q_index) *   // grad phi_i(x_q)
                                                  fe_face_values.shape_value(j, q_index)) * // phi_j(x_q)
                                                  fe_face_values.normal_vector(q_index) -   // n
                                                  gamma / (face->diameter()) *              // gamma/h
                                                  fe_face_values.shape_value(i, q_index) *  // phi_i(x_q)
                                                  fe_face_values.shape_value(j, q_index)) * // phi_j(x_q)  
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
  void HeatEquation<dim>::assemble_ode_explicit_part(const double t)
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
                                    QGauss<dim-1>(fe.degree + 1),
                                    update_values | update_gradients |
                                    update_quadrature_points | update_normal_vectors |
                                    update_JxW_values);

    // Set the cell-related objects
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
    Vector<double> cell_explicit_part(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Declare the right-hand side and boundary values objects
    RightHandSide<dim> right_hand_side;
    BoundaryValues<dim> boundary_values;

    // Set these objects' time to t
    right_hand_side.set_time(t);
    boundary_values.set_time(t);

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
            cell_explicit_part(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                      right_hand_side.value(x_q) *        // f(x_q)
                                      fe_values.JxW(q_index));            // dx
        }

        // Loop over faces at the boundary of the current cell (if there are any)
        for (auto &face: cell->face_iterators())
          if (face->at_boundary())
          {
            // Initialize the fe_face_values to current face
            fe_face_values.reinit(cell, face);

            // Loop over face quadrature points. The formula is a bit condensed,
            // in order to do less arithmetic operations.
            for (const unsigned int q_index : fe_face_values.quadrature_point_indices())
            {
              const auto &x_q = fe_face_values.quadrature_point(q_index);
              for (const unsigned int i : fe_face_values.dof_indices())
                cell_explicit_part(i) += ((gamma / (face->diameter()) *                 // gamma/h
                                          fe_face_values.shape_value(i, q_index) -    // phi_i(x_q)
                                          fe_face_values.shape_grad(i, q_index) *     // grad phi_i(x_q)
                                          fe_face_values.normal_vector(q_index)) *    // n
                                          boundary_values.value(x_q) *                // g(x_q)
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
  void HeatEquation<dim>::solve_ode()
  {
    // We set some additional data for the solver. In particular,
    // we want to exploit the fact that the mass matrix and the
    // Jacobian matrix are independent of time.
    const double out_prd = (final_time - initial_time)/nsteps;
    SUNDIALS::ARKode<Vector<double>>::AdditionalData data(initial_time,
                                                          final_time,
                                                          1e-2,     // Initial step size
                                                          out_prd,  // Output period
                                                          min_step, // Minimum step size
                                                          5,        // Maximum order
                                                          10,       // Maximum nonlinear iterations
                                                          true,     // implicit_function_is_linear
                                                          true,     // implicit_function_is_time_independent
                                                          true);    // mass_is_time_independent

    // Here we declare the ARKode object
    SUNDIALS::ARKode<Vector<double>> ode(data);

    // We have to tell the solver how to multiply the mass matrix
    // by a vector v
    ode.mass_times_vector = [&] (const double t,
                            const Vector<double> &v,
                            Vector<double> &Mv)
    {
      (void)t;
      mass_matrix.vmult(Mv, v);
    };
    
    // We set the explicit function
    ode.explicit_function = [&] (const double t,
                             const Vector<double> &y,
                             Vector<double> &explicit_f)
    {
      (void)y;
      assemble_ode_explicit_part(t);
      explicit_f = explicit_part;
    };

    // We set the implicit function
    ode.implicit_function = [&] (const double t,
                             const Vector<double> &y,
                             Vector<double> &implicit_f)
    {
      (void)t;
      jacobian_matrix.vmult(implicit_f,y);
    };

    // Since the implicit function is linear, we can supply
    // the Jacobian matrix directly to solve the implicit part
    // of the ODE with a single Newton iteration.
    ode.jacobian_times_vector = [&] (const Vector<double> &v,
                                  Vector<double> &Jv,
                                  const double t,
                                  const Vector<double> &y,
                                  const Vector<double> &fy)
    {
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
    ode.solve_mass = [&](SUNDIALS::SundialsOperator<Vector<double>> &op,
                        SUNDIALS::SundialsPreconditioner<Vector<double>> &prec,
                        Vector<double> &x,
                        const Vector<double> &b,
                        double tol)
    {
      // We forget about op and prec, since we want to provide
      // our custom preconditioner and solver 
      (void) op;
      (void) prec;
      
      // We provide the solver and the preconditioner in the
      // same way as we have done in previous tutorial programs
      SolverControl solver_control(1000, tol);
      SolverCG<Vector<double>> solver(solver_control);

      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(mass_matrix, 1.2);

      solver.solve(mass_matrix, x, b, preconditioner);
    };

    // QUESTION: is also the Jacobian matrix positive definite for
    // appropriate values of gamma? If so, we could use CG
    // also for the implicit part of the ODE.
    // However, I've encountered some problems in supplying the
    // ode.solve_linearized_system function, see temp_dummy.cc
    // in the directory other_files.

    // We supply a function for the output at fixed time intervals
    ode.output_step = [&](const double t,
                          const Vector<double> &sol,
                          const unsigned int step_number)
    {
      std::cout << "Time step " << step_number << " at t = "
                << t << "." << std::endl;
      std::cout << "l_inf norm of solution: " << solution.linfty_norm() << std::endl;

      output_results(sol, step_number);
    };

    // Finally, we are ready to solve the ODE
    ode.solve_ode(solution);
  }

 
 
  template <int dim>
  void HeatEquation<dim>::output_results(const Vector<double> &sol,
                                         const unsigned int step_no) const
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
  void HeatEquation<dim>::run()
  {
    // Mesh generation, DoF distribution and sparsity pattern allocation
     GridGenerator::hyper_L(triangulation);
    triangulation.refine_global(initial_global_refinement);
 
    setup_ode();

    // Assembly of the mass matrix and the implicit part of the ODE
    assemble_ode_matrices();

    // For a simpler test, we set the initial condition to zero
    VectorTools::interpolate(dof_handler,
                             Functions::ZeroFunction<dim>(),
                             solution);
 
    // ODE solution
    solve_ode();
  }
} // namespace nmpdeProject
 
 
int main()
{
  try
    {
      using namespace nmpdeProject;
 
      HeatEquation<2> heat_equation_solver;
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
