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

#include "project.h"

int
main(int argc, char *argv[])
{
  try
    {
      using namespace nmpdeProject;

      // Even if we don't run our code in parallel,
      // we still need to initialize MPI in order to use
      // our Trilinos objects.
      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      // If the program is run in parallel, throw an exception.
      AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                  ExcMessage(
                    "This program can only be run in serial, use ./project_" +
                    std::to_string(DEAL_DIMENSION) + "d or ./project_" +
                    std::to_string(DEAL_DIMENSION) + "d.g instead"));

      HeatParameters<DEAL_DIMENSION> par;
      HeatEquation<DEAL_DIMENSION>   heat_equation_solver(par);
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
