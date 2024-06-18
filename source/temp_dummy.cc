//-----------------------------------------------------------------------------------------------
// FIRST ATTEMPT AT PROVIDING A CUSTOM SOLVER FOR THE LINEARIZED SYSTEM
//-----------------------------------------------------------------------------------------------

    // First attempt at using a custom solver for the linearized system.
    // I'm not satisfied, because it's a hard-coded workaround.
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
      MyMatrix(SUNDIALS::SundialsOperator<Vector<double>> &op) : op(op) {}

      void vmult(Vector<double> &dst, const Vector<double> &src) const
      {
        op.vmult(dst, src);  // Use the matrix-vector product of the SundialsOperator
      }

    private:
      SUNDIALS::SundialsOperator<Vector<double>> &op;
    };

    // Now we use this class in the solve_linearized_system function
    ode.solve_linearized_system = [&](SUNDIALS::SundialsOperator<Vector<double>> &op,
                                      SUNDIALS::SundialsPreconditioner<Vector<double>> &prec,
                                      Vector<double> &x,
                                      const Vector<double> &b,
                                      double tol)
    {
      // We forget about prec, since we want to provide our custom preconditioner
      (void) prec;

      // Define the solver control and the CG solver
      SolverControl solver_control(1000, tol);
      SolverCG<Vector<double>> solver(solver_control);

      // Define the system matrix and reinit its sparsity pattern,
      // which is the same as the mass matrix and the Jacobian matrix
      MyMatrix sys_matrix(op);
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
    };

