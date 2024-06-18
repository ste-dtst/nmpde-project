/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "Template Deal.II Application", "index.html", [
    [ "Coding conventions used throughout deal.II", "http://www.dealii.org/developer/doxygen/deal.II/CodingConventions.html", null ],
    [ "When to use types::global_dof_index instead of unsigned int", "http://www.dealii.org/developer/doxygen/deal.II/GlobalDoFIndex.html", null ],
    [ "Glossary", "http://www.dealii.org/developer/doxygen/deal.II/DEALGlossary.html", null ],
    [ "Template instantiations", "http://www.dealii.org/developer/doxygen/deal.II/Instantiations.html", null ],
    [ "Changes between Version 1.0 and 2.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_1_0_and_2_0.html", null ],
    [ "Changes between Version 2.0 and 3.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_2_0_and_3_0.html", null ],
    [ "Changes between Version 3.0.0 and 3.0.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_0_0_and_3_0_1.html", null ],
    [ "Changes between Version 3.0.0 and 3.1.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_0_0_and_3_1_0.html", null ],
    [ "Changes between Version 3.1.0 and 3.1.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_1_0_and_3_1_1.html", null ],
    [ "Changes between Version 3.1.0 and 3.2.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_1_0_and_3_2_0.html", null ],
    [ "Changes between Version 3.1.1 and 3.1.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_1_1_and_3_1_2.html", null ],
    [ "Changes between Version 3.2.0 and 3.2.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_2_0_and_3_2_1.html", null ],
    [ "Changes between Version 3.2 and 3.3", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_2_and_3_3.html", null ],
    [ "Changes between Version 3.3.0 and 3.3.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_3_0_and_3_3_1.html", null ],
    [ "Changes between Version 3.3 and 3.4", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_3_and_3_4.html", null ],
    [ "Changes between Version 3.4 and 4.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_3_4_and_4_0.html", null ],
    [ "Changes between Version 4.0 and 5.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_4_0_and_5_0.html", null ],
    [ "Changes between Version 5.0 and 5.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_5_0_and_5_1.html", null ],
    [ "Changes between Version 5.1 and 5.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_5_1_and_5_2.html", null ],
    [ "Changes between Version 5.2 and 6.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_5_2_and_6_0.html", null ],
    [ "Changes between Version 6.0 and 6.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_6_0_and_6_1.html", null ],
    [ "Changes between Version 6.1 and 6.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_6_1_and_6_2.html", null ],
    [ "Changes between Version 6.2.0 and 6.2.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_6_2_0_and_6_2_1.html", null ],
    [ "Changes between Version 6.2 and 6.3", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_6_2_and_6_3.html", null ],
    [ "Changes between Version 6.3.0 and 6.3.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_6_3_0_and_6_3_1.html", null ],
    [ "Changes between Version 6.3 and 7.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_6_3_and_7_0.html", null ],
    [ "Changes between Version 7.0 and 7.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_7_0_and_7_1.html", null ],
    [ "Changes between Version 7.1 and 7.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_7_1_and_7_2.html", null ],
    [ "Changes between Version 7.2 and 7.3", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_7_2_and_7_3.html", null ],
    [ "Changes between Version 7.3 and 8.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_7_3_and_8_0.html", null ],
    [ "Changes between Version 8.0 and 8.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_0_and_8_1.html", null ],
    [ "Changes between Version 8.1 and 8.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_1_and_8_2.html", null ],
    [ "Changes between Version 8.2.0 and 8.2.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_2_0_and_8_2_1.html", null ],
    [ "Changes between Version 8.2.1 and 8.3", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_2_1_and_8_3.html", null ],
    [ "Changes between Version 8.3.0 and 8.4.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_3_and_8_4.html", null ],
    [ "Changes between Version 8.4.0 and 8.4.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_4_0_and_8_4_1.html", null ],
    [ "Changes between Version 8.4.1 and 8.4.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_4_1_and_8_4_2.html", null ],
    [ "Changes between Version 8.4.2 and 8.5.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_4_2_and_8_5_0.html", null ],
    [ "Changes between Version 8.5.0 and 9.0.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_8_5_0_and_9_0_0.html", null ],
    [ "Changes between Version 9.0.0 and 9.0.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_0_0_and_9_0_1.html", null ],
    [ "Changes between Version 9.0.1 and 9.1.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_0_1_and_9_1_0.html", null ],
    [ "Changes between Version 9.1.0 and 9.1.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_1_0_and_9_1_1.html", null ],
    [ "Changes between Version 9.1.1 and 9.2.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_1_1_and_9_2_0.html", null ],
    [ "Changes between Version 9.2.0 and 9.3.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_2_0_and_9_3_0.html", null ],
    [ "Changes between Version 9.3.0 and 9.3.1", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_3_0_and_9_3_1.html", null ],
    [ "Changes between Version 9.3.1 and 9.3.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_3_1_and_9_3_2.html", null ],
    [ "Changes between Version 9.3.2 and 9.3.3", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_3_2_and_9_3_3.html", null ],
    [ "Changes between Version 9.3.3 and 9.4.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_3_3_and_9_4_0.html", null ],
    [ "Changes between Version 9.4.0 and 9.5.0", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_4_0_and_9_5_0.html", null ],
    [ "Changes between Version 9.5.0 and 9.5.2", "http://www.dealii.org/developer/doxygen/deal.II/changes_between_9_5_0_and_9_5_2.html", null ],
    [ "Changes since the last release", "http://www.dealii.org/developer/doxygen/deal.II/recent_changes.html", null ],
    [ "Tutorial programs", "http://www.dealii.org/developer/doxygen/deal.II/Tutorial.html", null ],
    [ "The step-1 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_1.html", null ],
    [ "The step-10 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_10.html", null ],
    [ "The step-11 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_11.html", null ],
    [ "The step-12 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_12.html", null ],
    [ "The step-13 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_13.html", null ],
    [ "The step-14 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_14.html", null ],
    [ "The step-15 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_15.html", null ],
    [ "The step-16 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_16.html", null ],
    [ "The step-17 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_17.html", null ],
    [ "The step-18 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_18.html", null ],
    [ "The step-19 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_19.html", null ],
    [ "The step-2 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_2.html", null ],
    [ "The step-20 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_20.html", null ],
    [ "The step-21 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_21.html", null ],
    [ "The step-22 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_22.html", null ],
    [ "The step-23 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_23.html", null ],
    [ "The step-24 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_24.html", null ],
    [ "The step-25 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_25.html", null ],
    [ "The step-26 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_26.html", null ],
    [ "The step-27 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_27.html", null ],
    [ "The step-28 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_28.html", null ],
    [ "The step-29 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_29.html", null ],
    [ "The step-3 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_3.html", null ],
    [ "The step-30 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_30.html", null ],
    [ "The step-31 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_31.html", null ],
    [ "The step-32 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_32.html", null ],
    [ "The step-33 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_33.html", null ],
    [ "The step-34 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_34.html", null ],
    [ "The step-35 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_35.html", null ],
    [ "The step-36 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_36.html", null ],
    [ "The step-37 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_37.html", null ],
    [ "The step-38 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_38.html", null ],
    [ "The step-39 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_39.html", null ],
    [ "The step-4 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_4.html", null ],
    [ "The step-40 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_40.html", null ],
    [ "The step-41 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_41.html", null ],
    [ "The step-42 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_42.html", null ],
    [ "The step-43 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_43.html", null ],
    [ "The step-44 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_44.html", null ],
    [ "The step-45 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_45.html", null ],
    [ "The step-46 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_46.html", null ],
    [ "The step-47 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_47.html", null ],
    [ "The step-48 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_48.html", null ],
    [ "The step-49 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_49.html", null ],
    [ "The step-5 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_5.html", null ],
    [ "The step-50 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_50.html", null ],
    [ "The step-51 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_51.html", null ],
    [ "The step-52 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_52.html", null ],
    [ "The step-53 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_53.html", null ],
    [ "The step-54 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_54.html", null ],
    [ "The step-55 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_55.html", null ],
    [ "The step-56 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_56.html", null ],
    [ "The step-57 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_57.html", null ],
    [ "The step-58 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_58.html", null ],
    [ "The step-59 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_59.html", null ],
    [ "The step-6 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_6.html", null ],
    [ "The step-60 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_60.html", null ],
    [ "The step-61 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_61.html", null ],
    [ "The step-62 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_62.html", null ],
    [ "The step-63 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_63.html", null ],
    [ "The step-64 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_64.html", null ],
    [ "The step-65 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_65.html", null ],
    [ "The step-66 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_66.html", null ],
    [ "The step-67 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_67.html", null ],
    [ "The step-68 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_68.html", null ],
    [ "The step-69 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_69.html", null ],
    [ "The step-7 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_7.html", null ],
    [ "The step-70 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_70.html", null ],
    [ "The step-71 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_71.html", null ],
    [ "The step-72 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_72.html", null ],
    [ "The step-74 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_74.html", null ],
    [ "The step-75 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_75.html", null ],
    [ "The step-76 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_76.html", null ],
    [ "The step-77 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_77.html", null ],
    [ "The step-78 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_78.html", null ],
    [ "The step-79 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_79.html", null ],
    [ "The step-8 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_8.html", null ],
    [ "The step-81 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_81.html", null ],
    [ "The step-82 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_82.html", null ],
    [ "The step-85 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_85.html", null ],
    [ "The step-87 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_87.html", null ],
    [ "The step-89 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_89.html", null ],
    [ "The step-9 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_9.html", null ],
    [ "The step-90 tutorial program", "http://www.dealii.org/developer/doxygen/deal.II/step_90.html", null ],
    [ "The 'Viscoelastoplastic topography evolution' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_CeresFE.html", null ],
    [ "The 'Distributed LDG Method' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Distributed_LDG_Method.html", null ],
    [ "The 'Distributed moving laser heating' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Distributed_Moving_Laser_Heating.html", null ],
    [ "The 'Elastoplastic Torsion' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_ElastoplasticTorsion.html", null ],
    [ "The 'Linear Elastic Active Skeletal Muscle Model' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Linear_Elastic_Active_Skeletal_Muscle_Model.html", null ],
    [ "The 'MCMC for the Laplace equation' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_MCMC_Laplace.html", null ],
    [ "The 'Goal-Oriented hp-Adaptivity for the Maxwell Eigenvalue Problem' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Maxwell_Eigenvalue-hp-Refinement.html", null ],
    [ "The 'Higher Order Multipoint Flux Mixed Finite Element (MFMFE) methods' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_MultipointFluxMixedFiniteElementMethods.html", null ],
    [ "The 'TRBDF2-DG projection solver for the incompressible Navier-Stokes equations' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_NavierStokes_TRBDF2_DG.html", null ],
    [ "The 'Nonlinear poro-viscoelasticity' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Nonlinear_PoroViscoelasticity.html", null ],
    [ "The 'Quasi-Static Finite-Strain Compressible Elasticity' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Quasi_static_Finite_strain_Compressible_Elasticity.html", null ],
    [ "The 'Quasi-Static Finite-Strain Quasi-incompressible Visco-elasticity' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Quasi_static_Finite_strain_Quasi_incompressible_ViscoElasticity.html", null ],
    [ "The 'Generalized Swift-Hohenberg Equation Solve' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_Swift_Hohenberg-Solver.html", null ],
    [ "The 'Adaptive advection-reaction' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_advection_reaction_estimator.html", null ],
    [ "The 'Convection Diffusion Reaction' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_cdr.html", null ],
    [ "The 'Laplace equation coupled to an external simulation program' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_coupled_laplace_problem.html", null ],
    [ "The 'Goal-oriented mesh adaptivity in elastoplasticity problems' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_goal_oriented_elastoplasticity.html", null ],
    [ "The 'Information density-based mesh refinement' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_information_based_mesh_refinement.html", null ],
    [ "The 'Parallel in Time Heat Equation' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_parallel_in_time.html", null ],
    [ "The 'Time-dependent Navier-Stokes' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_time_dependent_navier_stokes.html", null ],
    [ "The 'Two phase flow interaction ' code gallery program", "http://www.dealii.org/developer/doxygen/deal.II/code_gallery_two_phase_flow.html", null ],
    [ "The deal.II code gallery", "http://www.dealii.org/developer/doxygen/deal.II/CodeGallery.html", null ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", null ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"CodeGallery.html",
"step_8.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';