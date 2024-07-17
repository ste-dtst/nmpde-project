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


// We start including the header file of the project
#include "project.h"

// We include some other standard deal.II headers
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

// We include the MeshWorker functionality in order to parallelize
// the assembly of the matrices and vectors for the ODE
#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

// This will be needed to integrate in time
#include <deal.II/sundials/arkode.h>

// In order to ask ARKode for some info on the time stepping,
// we need to include some functionalities that are not
// wrapped by deal.II: we need the ARKStep header itself
#include <arkode/arkode_arkstep.h>

// Finally, we import the custom namespace we have created
using namespace nmpdeProject;



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
  // This function is lighter than usual. Normally, here we would
  // (re)initialize also the matrices, but this would waste some
  // computational resources in case we wanted to refine the mesh
  // multiple times before continuing with the time integration.
  // In fact, the error estimation is "matrices-independent".
  // Hence, we move the matrix initialization in the function
  // assemble_ode_matrices.

  // As usual, we distribute the degrees of freedom
  dof_handler.distribute_dofs(fe);

  std::cout << "===========================================" << std::endl
            << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  // We reinitialize the constraints object
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  // Finally, we initialize the solution vector
  solution.reinit(dof_handler.n_dofs());
}



template <int dim>
void
HeatEquation<dim>::assemble_ode_matrices()
{
  // This function will be called every time the matrices
  // need to be updated, in particular after a (chain of)
  // mesh refinement(s). Here we do either the setup and
  // the assembly of the matrices.

  // We allocate the sparsity pattern for the matrices
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  // Finally, we initialize the matrices
  mass_matrix.reinit(sparsity_pattern);
  jacobian_matrix.reinit(sparsity_pattern);

  // We now assemble the mass matrix and the matrix for the
  // implicit part of the ODE using the MeshWorker framework.
  // First, we need the ScratchData and CopyData objects.

  // Scratch data: declare the quadrature formulas
  MeshWorker::ScratchData<dim> scratch_data(fe,
                                            QGauss<dim>(fe.degree + 1),
                                            update_values | update_gradients |
                                              update_quadrature_points |
                                              update_JxW_values,
                                            QGauss<dim - 1>(fe.degree + 1),
                                            update_values | update_gradients |
                                              update_quadrature_points |
                                              update_normal_vectors |
                                              update_JxW_values);

  // Copy data: we need two matrices to store the local contributions
  // to the mass matrix and the jacobian matrix, no vector.
  MeshWorker::CopyData<2, 0, 1> copy_data(fe.n_dofs_per_cell());

  // Cell worker
  const auto cell_worker = [&](const auto &cell, auto &scratch, auto &copy) {
    // Initialize the fe_values to current cell
    const auto &fe_values = scratch.reinit(cell);

    // Get the cell matrices, cell explicit part, and local dof indices
    auto &cell_mass_matrix     = copy.matrices[0];
    auto &cell_jacobian_matrix = copy.matrices[1];
    auto &local_dof_indices    = copy.local_dof_indices[0];

    // Initialize the matrices to zero
    cell_mass_matrix     = 0;
    cell_jacobian_matrix = 0;

    // Initialize the local dof indices to current cell
    cell->get_dof_indices(local_dof_indices);

    // The usual loop over quadrature points, with a slight variation.
    // Since the matrices are symmetric, we do the calculations only
    // for the lower triangular part and save some computational work.
    for (const unsigned int q_index : fe_values.quadrature_point_indices())
      for (const unsigned int i : fe_values.dof_indices())
        for (unsigned int j = 0; j <= i; ++j)
          {
            // Do the calculations
            auto mass_temp = (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                              fe_values.shape_value(j, q_index) * // phi_j(x_q)
                              fe_values.JxW(q_index));            // dx

            auto jacobian_temp =
              (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
               fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
               fe_values.JxW(q_index));           // dx

            // Update the matrices exploiting the symmetry
            if (j == i)
              {
                cell_mass_matrix(i, j) += mass_temp;
                cell_jacobian_matrix(i, j) -= jacobian_temp;
              }
            else // j < i
              {
                cell_mass_matrix(i, j) += mass_temp;
                cell_mass_matrix(j, i) += mass_temp;
                cell_jacobian_matrix(i, j) -= jacobian_temp;
                cell_jacobian_matrix(j, i) -= jacobian_temp;
              }
          }
  };

  // Boundary worker
  const auto boundary_worker =
    [&](const auto &cell, const auto &face_no, auto &scratch, auto &copy) {
      // Initialize the fe_face_values to current face
      auto &fe_face_values = scratch.reinit(cell, face_no);

      // In the boundary worker we will work only for the jacobian matrix,
      // therefore we create an alias only for it. We DO NOT reinitialize
      // anything here, or we will overwrite data computed in the cell worker.
      auto &cell_jacobian_matrix = copy.matrices[1];

      // Since we already have initialized the local dof indices in the cell
      // worker, we don't need to do it again here.

      // Compute the diameter of the face, being aware that
      // for dim = 1 it is not defined (-> we set it to 1).
      // Here we use a constexpr if statement (C++17).
      double face_diam;
      if constexpr (dim == 1)
        face_diam = 1;
      else
        face_diam = cell->face(face_no)->diameter();

      // Loop over face quadrature points. The formulas are a bit
      // condensed, in order to do less arithmetic operations.
      // As before, we also exploit the symmetry of the matrices.
      for (const unsigned int q_index :
           fe_face_values.quadrature_point_indices())
        for (const unsigned int i : fe_face_values.dof_indices())
          for (unsigned int j = 0; j <= i; ++j)
            {
              // Do the calculations
              auto jacobian_temp =
                (((fe_face_values.shape_value(i, q_index) *  // phi_i(x_q)
                     fe_face_values.shape_grad(j, q_index) + // grad phi_j(x_q)
                   fe_face_values.shape_grad(i, q_index) *   // grad phi_i(x_q)
                     fe_face_values.shape_value(j, q_index)) * // phi_j(x_q)
                    fe_face_values.normal_vector(q_index) -    // n
                  par.gamma / face_diam *                      // gamma/h
                    fe_face_values.shape_value(i, q_index) *   // phi_i(x_q)
                    fe_face_values.shape_value(j, q_index)) *  // phi_j(x_q)
                 fe_face_values.JxW(q_index));                 // dx

              // Update the matrix exploiting the symmetry
              if (j == i)
                cell_jacobian_matrix(i, j) += jacobian_temp;
              else // j < i
                {
                  cell_jacobian_matrix(i, j) += jacobian_temp;
                  cell_jacobian_matrix(j, i) += jacobian_temp;
                }
            }
    };

  // Copier function
  const auto copier = [&](const auto &copy) {
    // auto &cell_mass_matrix     = copy.matrices[0];
    // auto &cell_jacobian_matrix = copy.matrices[1];
    // auto &local_dof_indices    = copy.local_dof_indices[0];

    // Distribute local to global
    constraints.distribute_local_to_global(copy.matrices[0],
                                           copy.local_dof_indices[0],
                                           mass_matrix);
    constraints.distribute_local_to_global(copy.matrices[1],
                                           copy.local_dof_indices[0],
                                           jacobian_matrix);
  };

  // Finally, we run the MeshWorker loop
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces,
                        boundary_worker);
}



template <int dim>
void
HeatEquation<dim>::assemble_ode_explicit_part(const double t)
{
  // We assemble also the explicit part of the ODE with MeshWorker.

  // First, we reinit the global vector to zero
  explicit_part.reinit(dof_handler.n_dofs());

  // Set the right-hand side and boundary value's time to t
  par.right_hand_side.set_time(t);
  par.boundary_value.set_time(t);

  // Scratch data: declare the quadrature formulas
  MeshWorker::ScratchData<dim> scratch_data(fe,
                                            QGauss<dim>(fe.degree + 1),
                                            update_values | update_gradients |
                                              update_quadrature_points |
                                              update_JxW_values,
                                            QGauss<dim - 1>(fe.degree + 1),
                                            update_values | update_gradients |
                                              update_quadrature_points |
                                              update_normal_vectors |
                                              update_JxW_values);

  // Copy data: we need only a vector to store the local contributions
  // to the explicit part of the ODE.
  MeshWorker::CopyData<0, 1, 1> copy_data(fe.n_dofs_per_cell());

  // Cell worker
  const auto cell_worker = [&](const auto &cell, auto &scratch, auto &copy) {
    // Initialize the fe_values to current cell
    const auto &fe_values = scratch.reinit(cell);

    // Get the cell explicit part and the local dof indices
    auto &cell_explicit_part = copy.vectors[0];
    auto &local_dof_indices  = copy.local_dof_indices[0];

    // Initialize the vector to zero
    cell_explicit_part = 0;

    // Initialize the local dof indices to current cell
    cell->get_dof_indices(local_dof_indices);

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
  };

  // Boundary worker
  const auto boundary_worker =
    [&](const auto &cell, const auto &face_no, auto &scratch, auto &copy) {
      // Initialize the fe_face_values to current face
      auto &fe_face_values = scratch.reinit(cell, face_no);

      // As done before, we DO NOT reinitialize anything here, or
      // we will overwrite data computed in the cell worker.
      auto &cell_explicit_part = copy.vectors[0];

      // Since we already have initialized the local dof indices in the cell
      // worker, we don't need to do it again here.

      // Compute the diameter of the face, being aware that
      // for dim = 1 it is not defined (-> we set it to 1).
      // Here we use a constexpr if statement (C++17).
      double face_diam;
      if constexpr (dim == 1)
        face_diam = 1;
      else
        face_diam = cell->face(face_no)->diameter();

      // Loop over face quadrature points. The formula is a bit
      // condensed, in order to do less arithmetic operations.
      for (const unsigned int q_index :
           fe_face_values.quadrature_point_indices())
        {
          const auto &x_q = fe_face_values.quadrature_point(q_index);
          for (const unsigned int i : fe_face_values.dof_indices())
            cell_explicit_part(i) +=
              ((par.gamma / face_diam *                    // gamma/h
                  fe_face_values.shape_value(i, q_index) - // phi_i(x_q)
                fe_face_values.shape_grad(i, q_index) *    // grad phi_i(x_q)
                  fe_face_values.normal_vector(q_index)) * // n
               par.boundary_value.value(x_q) *             // g(x_q)
               fe_face_values.JxW(q_index));               // dx
        }
    };

  // Copier function
  const auto copier = [&](const auto &copy) {
    // auto &cell_explicit_part = copy.vectors[0];
    // auto &local_dof_indices  = copy.local_dof_indices[0];

    // Distribute local to global
    constraints.distribute_local_to_global(copy.vectors[0],
                                           copy.local_dof_indices[0],
                                           explicit_part);
  };

  // Finally, we run the MeshWorker loop
  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces,
                        boundary_worker);
}



template <int dim>
void
HeatEquation<dim>::refine_mesh(Vector<double>    &sol,
                               const unsigned int min_grid_level,
                               const unsigned int max_grid_level)
{
  // If we want to refine the mesh, we have to update the dimension
  // of matrices and vectors accordingly, update mass_matrix and
  // jacobian_matrix, transfer the solution to the new mesh.
  // This function takes care of the mesh and of the solution.
  // In particular, the solution vector to be transferred is the given
  // sol instead of solution, because the refine_mesh routine is
  // called in ode.solver_should_restart (check further on).

  // We mark the cells that have to be refined or coarsed
  GridRefinement::refine_and_coarsen_fixed_fraction(
    triangulation,
    estimated_error_per_cell,
    par.refinement_top_fraction,
    par.refinement_bottom_fraction);

  // We clear the flags that would make the mesh too coarse or too fine
  if (triangulation.n_levels() > max_grid_level)
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(max_grid_level))
      cell->clear_refine_flag();
  for (const auto &cell :
       triangulation.active_cell_iterators_on_level(min_grid_level))
    cell->clear_coarsen_flag();

  // The following step is to transfer the solution to the new mesh
  SolutionTransfer<dim> solution_trans(dof_handler);

  Vector<double> previous_solution;
  previous_solution = sol;
  triangulation.prepare_coarsening_and_refinement();
  solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

  triangulation.execute_coarsening_and_refinement();
  setup_ode();

  solution_trans.interpolate(previous_solution, sol);
  constraints.distribute(sol);
}



template <int dim>
void
HeatEquation<dim>::solve_ode()
{
  // Here we setup the ARKode solver and solve the ODE.
  // What we will do below depends also on the refinement
  // strategy we want to implement. In particular, we'd like
  // to be able to refine the mesh after each solution output
  // and also at additional timesteps between outputs. Our solver
  // can be provided two functions to handle output and refinement:
  // output_step and solver_should_restart. Unfortunately,
  // they both are called after the same amount of time.
  // Therefore, we choose to provide only solver_should_restart
  // and have a loop in which we will call solve_ode_incrementally
  // and output the solution. SUNDIALS will take care of
  // calling solver_should_restart between outputs, as desired.

  // At first, we have to fix the time period between two
  // solution outputs and the one between two error estimations.
  const double out_prd = (par.final_time - par.initial_time) / par.out_steps;
  const double est_prd = out_prd / par.est_steps;

  // We set some additional data for the solver. In particular,
  // we want to exploit the fact that the mass matrix and the
  // Jacobian matrix are independent of time.
  SUNDIALS::ARKode<Vector<double>>::AdditionalData data(
    par.initial_time,        // Initial time
    par.final_time,          // Final time
    par.initial_step_size,   // Initial step size
    est_prd,                 // Period between two error estimations (!)
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
                       double                                            tol) {
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

    constraints.distribute(x);
  };

  // Also the matrix of the linearized system is
  // positive definite for any legal choice of gamma.
  // Therefore, we should use CG also for the implicit part of the ODE.

  // First attempt at using a custom solver for the linearized system.
  // I'm not satisfied, because it's a hard-coded workaround.
  //
  // ----to be improved, using it to test adaptive mesh refinement----
  //
  // Since the SundialsOperator in ode.solve_linearized_system will only know
  // how to do matrix-vector products, we need to define a custom class
  // that represents the matrix. This class will inherit from
  // SparseMatrix<double> and will implement a virtual method for
  // vmult, which is necessary for the SolverCG class, based on the
  // SundialsOperator one.
  class MyMatrix : public SparseMatrix<double>
  {
  public:
    MyMatrix(SUNDIALS::SundialsOperator<Vector<double>> &op)
      : op(op)
    {}

    void
    vmult(Vector<double> &dst, const Vector<double> &src) const
    {
      op.vmult(dst,
               src); // Use the matrix-vector product of the SundialsOperator
    }

  private:
    SUNDIALS::SundialsOperator<Vector<double>> &op;
  };

  // Now we use this class in the solve_linearized_system function
  ode.solve_linearized_system =
    [&](SUNDIALS::SundialsOperator<Vector<double>>       &op,
        SUNDIALS::SundialsPreconditioner<Vector<double>> &prec,
        Vector<double>                                   &x,
        const Vector<double>                             &b,
        double                                            tol) {
      // We forget about prec, since we want to provide our custom
      // preconditioner
      (void)prec;

      // Define the solver control and the CG solver
      SolverControl            solver_control(1000, tol);
      SolverCG<Vector<double>> solver(solver_control);

      // Define the system matrix and reinit its sparsity pattern,
      // which is the same as the mass matrix and the Jacobian matrix
      MyMatrix        sys_matrix(op);
      SparsityPattern sparsity_pattern;
      sparsity_pattern.copy_from(mass_matrix.get_sparsity_pattern());
      sys_matrix.reinit(sparsity_pattern);

      // PROBLEM: the preconditioner cannot be initialized at this point,
      // because we only know how sys_matrix acts for products... in
      // particular, at the moment it is a sparse zero matrix.
      // Is there a workaround for this, other than providing a
      // jacobian_preconditioner_solve function?

      // PreconditionSSOR<> preconditioner;
      // preconditioner.initialize(matrix, 1.2);

      // Solve the system with no preconditioner
      solver.solve(sys_matrix, x, b, PreconditionIdentity());

      constraints.distribute(x);
    };

  // Here we store some relevant info on the time stepping
  double   inter_time = par.initial_time; // Intermediate time
  double   curr_time  = par.initial_time; // Current time
  long int nsteps     = 0;                // Number of steps taken in the solver
  double   hlast =
    par.initial_step_size; // Step size taken on the last internal step

  // Each reset of the solver will also reset the counter of the
  // number of steps in the internal ARKode memory, therefore we need
  // an auxiliary variable to store the number of steps properly.
  // The update of nsteps will be done in solver_should_restart.
  long int nsteps_temp = 0;

  // With the following function, we decide whether a mesh refinement
  // is needed at time t. In that case, we call refine_mesh() and
  // return true in order to trigger the reset() function of the
  // ARKode solver. Otherwise, we return false.
  // NOTA BENE: solver_should_restart is called recursively until it
  // returns false, so we can refine the mesh multiple times, but also
  // get stuck in a loop if we never meet the condition for returning false.
  // Moreover, the matrices are not involved in the error estimation,
  // hence we can update them only after the last mesh refinement.
  ode.solver_should_restart = [&](const double t, Vector<double> &sol) {
    // We do the Kelly error estimation
    estimated_error_per_cell.reinit(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      sol,
      estimated_error_per_cell);

    // Each output step comes right after a restart check.
    // Therefore, we have to update hlast here, otherwise we would
    // get 0 when the output step comes right after a refinement.
    // It is a bit of an overkill, since we update it more times
    // than needed, however the computational effort is negligible.
    if (consecutive_refinements == 0)
      ARKStepGetLastStep(ode.get_arkode_memory(), &hlast);
    // else hlast is already updated

    // We check if the error is too large.
    if (estimated_error_per_cell.l2_norm() > par.refinement_threshold)
      {
        if (consecutive_refinements == 5)
          {
            // If we have already done 5 consecutive refinements,
            // we stop the process and return false.
            std::cout << "==========================================="
                      << std::endl
                      << "5 refinements were not sufficient at time " << t
                      << "." << std::endl
                      << "l2 norm of the error estimator: "
                      << estimated_error_per_cell.l2_norm() << std::endl;

            consecutive_refinements = 0; // Reset the counter
            assemble_ode_matrices();     // Do update the matrices

            return false;
          }
        else
          {
            // Otherwise, we refine the mesh and increase the counter
            refine_mesh(sol, par.min_refinement, par.max_refinement);
            ++consecutive_refinements;

            // Also, we update nsteps before the memory gets reset
            ARKStepGetNumSteps(ode.get_arkode_memory(), &nsteps_temp);
            nsteps += nsteps_temp;

            return true;
          }
      }
    else
      {
        // If the error is small enough, we reset the counter
        if (consecutive_refinements > 0)
          {
            std::cout << "==========================================="
                      << std::endl
                      << "Done " << consecutive_refinements
                      << " refinement(s) at time " << t << "." << std::endl
                      << "l2 norm of the error estimator: "
                      << estimated_error_per_cell.l2_norm() << std::endl;

            consecutive_refinements = 0; // Reset the counter
            assemble_ode_matrices();     // Do update the matrices

            return false;
          }
        else // If the error is small and the counter is 0, we just return false
          return false;
      }
  };

  // Finally, we are ready to solve the ODE
  for (unsigned int step = 1; step <= par.out_steps; ++step)
    {
      // Solve the ODE until inter_time
      inter_time += out_prd;
      if (step == par.out_steps)
        // At the last step, we may get past par.final_time
        // due to rounding errors. We have to avoid that.
        inter_time = std::min(inter_time, par.final_time);
      ode.solve_ode_incrementally(solution, inter_time);

      // Give some info to the user
      ARKStepGetCurrentTime(ode.get_arkode_memory(), &curr_time);
      std::cout << "-------------------------------------------" << std::endl;
      std::cout << "Time step " << step << " at t = " << curr_time << "."
                << std::endl;

      // Some info on the time stepping so far.
      // Note: since nsteps is updated to the last reset, the total
      // number of steps taken so far is nsteps + a number that
      // will be stored in nsteps_temp.
      ARKStepGetNumSteps(ode.get_arkode_memory(), &nsteps_temp);
      std::cout << "Number of ARKode steps taken so far: "
                << nsteps + nsteps_temp << std::endl
                << "Step size taken on the last internal step: " << hlast
                << std::endl;

      // Some info on the last error estimation
      std::cout << "l2 norm of the error estimator: "
                << estimated_error_per_cell.l2_norm() << std::endl;

      // If the exact solution is known, we compute the L2 error
      // at current time
      if (par.sol_is_known)
        {
          par.exact_solution.set_time(curr_time);
          Vector<double> error(solution.size());
          VectorTools::integrate_difference(dof_handler,
                                            solution,
                                            par.exact_solution,
                                            error,
                                            QGauss<dim>(fe.degree + 1),
                                            VectorTools::L2_norm);
          std::cout << "L2 error on the solution: " << error.l2_norm()
                    << std::endl;
        }

      // We output the solution at the current time step
      output_results(solution, step);
    }
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

  const std::string dirname = "output_" + std::to_string(dim) + "d/";
  const std::string filename =
    "solution-" + Utilities::int_to_string(step_no, 3) + ".vtu";
  std::ofstream output(dirname + filename);
  data_out.write_vtu(output);

  static std::vector<std::pair<double, std::string>> times_and_names;
  times_and_names.push_back({step_no, filename});

  std::ofstream pvd_output(dirname + "solution.pvd");

  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}



template <int dim>
void
HeatEquation<dim>::run()
{
  // Mesh generation, DoF distribution and constraints setup
  if (dim == 1)
    GridGenerator::hyper_cube(triangulation, -1, 1);
  else
    GridGenerator::hyper_L(triangulation);
  triangulation.refine_global(par.initial_refinement);

  setup_ode();

  // Assembly of the mass matrix and the implicit part of the ODE
  assemble_ode_matrices();

  // We set the initial condition and output it
  VectorTools::interpolate(dof_handler, par.initial_value, solution);
  constraints.distribute(solution);
  output_results(solution, 0);

  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Time step 0 at t = " << par.initial_time << "." << std::endl;

  // If the exact solution is known, we compute the L2 error
  // at start time
  if (par.sol_is_known)
    {
      par.exact_solution.set_time(par.initial_time);
      Vector<double> error(solution.size());
      VectorTools::integrate_difference(dof_handler,
                                        solution,
                                        par.exact_solution,
                                        error,
                                        QGauss<dim>(fe.degree + 1),
                                        VectorTools::L2_norm);
      std::cout << "L2 error: " << error.l2_norm() << std::endl;
    }

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


template class nmpdeProject::HeatEquation<1>;
template class nmpdeProject::HeatEquation<2>;
template class nmpdeProject::HeatEquation<3>;