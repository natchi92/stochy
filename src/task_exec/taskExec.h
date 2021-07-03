/*
 * taskExec.h
 *
 *  Created on: 19 Feb 2018
 *      Author: nathalie
 */

#ifndef TASKEXEC_H_
#define TASKEXEC_H_

#include "Bmdp.h"
#include "FAUST.h"
#include "InputSpec.h"

static void performTask(inputSpec_t<arma::mat, int> &input) const {
  switch (input.myTask.task) {
  case 1: // Perform simulation depending on model type
  {

    if (input.myTask.T <= 0) {
      std::cout << "Incorrect length of time horizon, value needs to be "
                   "positive and greater then 0";
      exit(0);
    }
    if (input.myTask.runs <= 0) {
      std::cout << "Incorrect number of monte carlo simulations, value needs "
                   "to be positive and greater then 0"
                << std::endl;
      std::cout << "Will proceed with default number of simulations i.e. 5000 "
                   "simulation runs";
      input.myTask.runs = 5000;
    }

    // Check model
    int num_cont = input.myModel.n;
    for (int i = 0; i < input.myModel.Q; ++i) {
      if (input.myModel.x_mod[i].A.n_rows != num_cont) {
        std::cout << "Different number of continuous variables for each mode, "
                     "currently not supported. Work in progress"
                  << std::endl;
        exit(0);
      }
      if (input.myModel.x_mod[i].sigma.n_rows != num_cont) {
        std::cout << "Different number of continuous variables for each mode, "
                     "currently not supported. Work in progress"
                  << std::endl;
        exit(0);
      }
    }
    if (input.myModel.Tq.n_rows != input.myModel.x_mod.size()) {
      std::cout << "Incorrect transition matrix. Need to have number of rows "
                   "== number of columns == number of models representing the "
                   "evolution of the continuous variables."
                << std::endl;
      exit(0);
    }
    //TODO(ncauchi) are the inputs required here?
    input.myModel.run(input.myModel, input.myTask.T, input.myTask.runs);

    break;
  }
  case 2: // Perform verification task depending on model and tool to be
          // interfaced with
  {
    // faust_t myF;
    // shs_t<arma::mat, int> myModel = input.myModel;
    // myF.model = myModel;
    try {
      // Check if dynamics are ill-scaled
      // Obtaining problem definition
      clock_t begin, end; //TODO(ncauchi) change to use std::chrono
      begin = clock();
      double time = 0;

      int Problem = 100 * input.myTask.propertySpec;
      int Gridding = 10 * input.myTask.typeGrid;
      int Distribution = input.myTask.assumptionsKernel;
      int Control = input.myTask.Controlled;

      // Problem variablesmyF
      double epsilon = input.myTask.eps;
      int N = input.myTask.T;
      // shs_t<arma::mat, int> model = input.myModel;

      // Get Safe, Input and Target set
      arma::mat SafeSet = input.myTask.safeSet;
      arma::mat InputSet = input.myTask.inputSet;
      arma::mat TargetSet = input.myTask.targetSet;

      faust_t taskFAUST = myF; //TODO(ncauchi) what is the difference between faust_t and the input model; do better calling

      // Check correctness of SafeSet input
      if (SafeSet.n_cols != 2) {
        std::cout << "There is no correctly defined Safe Set";
        exit(0);
      }
      int temp = static_cast<int>(arma::accu((SafeSet.col(1) - SafeSet.col(0)) < 0));
      if (temp) {
        std::cout << "The edges of the Safe Set must have positive length. "
                     "Make sure that the first column is the lower bound and "
                     "the second column is the upper bound";
        exit(0);
      }
      // Check if Safe Set needs rescaling
      // Check if need to rescale boundary
      arma::mat max_Ss = arma::max(SafeSet);
      arma::mat min_Ss = arma::min(SafeSet);
      double diff_Ss = max_Ss(1) - min_Ss(0);
      arma::mat j = arma::eye<arma::mat>(SafeSet.n_rows, SafeSet.n_rows);
      if (diff_Ss > 50) {
        // Identify row with smallest values to rescale to that
        arma::mat min_row = arma::min(SafeSet, 0);
        arma::mat OrSS = SafeSet; // The original Safeset definition
        arma::mat to_inv = OrSS;
        for (unsigned i = 0; i < OrSS.n_rows; i++) {
          if (min_row(0, 1) != SafeSet(i, 1)) {
            SafeSet.row(i) = min_row;
          }
        }
        arma::mat y_inv = arma::diagmat(min_row);
        arma::mat j_bar = OrSS / y_inv;
        j = j_bar.replace(arma::datum::inf, 0);
        for (unsigned k = 0; k < model.x_mod.size(); k++) {
          taskFAUST.model.x_mod[k].sigma = arma::inv(j) * model.x_mod[k].sigma;
        }
      }
      if (Control == 0) {
        InputSet.clear();
      } else {
        if (model.x_mod[0].B.n_cols != InputSet.n_rows) {
          std::cout << "There is no correctly defined Input Set";
          exit(0);
        }
      }

      // Check if dimensions are correct
      if ((unsigned)model.x_mod[0].A.n_cols != SafeSet.n_rows) {
        std::cout << "The dimension of the Kernel does not match the "
                     "dimensions of the Safe Set";
        exit(0);
      }

      // Check correctness of target set if problem of reach and avoid
      if (Problem == 2) {
        if (TargetSet.n_cols != 2) {
          std::cout << "There is no correctly defined Target Set";
          exit(0);
        }
        double temp = arma::accu((TargetSet.col(1) - TargetSet.col(0)) < 0);
        if (temp > 0) {
          std::cout << "The edges of the Target Set must have positive length. "
                       "Make sure that the first column is the lower bound and "
                       "the second column is the upper bound";
          exit(0);
        }
        // Check if Target Set is inside Safe Set
        arma::umat uv = ((SafeSet - TargetSet) > 0);
        arma::umat temp1 = uv;
        arma::umat temp2 = arma::zeros<arma::umat>(SafeSet.n_rows, 1);
        temp2 =
            arma::join_horiz(temp2, arma::ones<arma::umat>(SafeSet.n_rows, 1));
        if (!arma::approx_equal(temp1, temp2, "both", 2, 0.1)) {
          arma::umat temp3 = (((SafeSet - TargetSet) >= 0));
          if (!arma::approx_equal(temp3, temp2, "both", 2, 0.1)) {
            std::cout << "The Target Set cannot be outside the Safe Set";
            exit(0);
          }
        }
      }

      // Solving the problem
      // Because of taking the center point, the error will be twice as small.
      // This allows to make epsilon twice as large.
      epsilon = 2 * epsilon;
      if (epsilon <= 0) {
        std::cout << "Incorrect maximum abstraction error, needs to be > 0"
                  << std::endl;
        exit(0);
      }
      int task2Solve = Problem + Gridding +
                       Distribution; // To avoid using list of if-condition
                                     // statements since it is slower
      // For multiple modes store the Tq for each mode in a vector
      std::vector<std::vector<arma::mat>> Tp_all;
      for (size_t i = 0; i < input.myModel.Q; i++) {
        // Get current model to perform abstraction and task on
        input.myModel.x_mod[0] = input.myModel.x_mod[i];
        taskFAUST.myKernel(input.myModel);
        switch (Control) {
        case 0: {
          switch (task2Solve) {
          case 111: {
            taskFAUST.Uniform_grid(epsilon, N, SafeSet);
            break;
          }
          case 121: {
            taskFAUST.Uniform_grid(10 * epsilon, N, SafeSet);
            taskFAUST.Adaptive_grid_multicell(epsilon, N);
            break;
          }
          case 131: {
            taskFAUST.Uniform_grid(10 * epsilon, N, SafeSet);
            taskFAUST.Adaptive_grid_multicell_semilocal(epsilon, N, SafeSet);
            break;
          }
          case 211: {
            taskFAUST.Uniform_grid_ReachAvoid(epsilon, N, SafeSet, TargetSet);
            break;
          }
          case 221: {
            taskFAUST.Uniform_grid_ReachAvoid(10 * epsilon, N, SafeSet,
                                              TargetSet);
            taskFAUST.Adaptive_grid_ReachAvoid(epsilon, N, SafeSet, TargetSet);
            break;
          }
          case 231: {
            taskFAUST.Uniform_grid_ReachAvoid(10 * epsilon, N, SafeSet,
                                              TargetSet);
            taskFAUST.Adaptive_grid_ReachAvoid_semilocal(epsilon, N, SafeSet,
                                                         TargetSet);
            break;
          }
          case 112: {
            taskFAUST.Uniform_grid_MCapprox(epsilon, N, SafeSet);
            break;
          }
          case 212: {
            taskFAUST.Uniform_grid_ReachAvoid_MCapprox(epsilon, N, SafeSet,
                                                       TargetSet);
            break;
          }
          case 122: {
            std::cout << "This options is not available yet. Work in progress.";
            taskFAUST.Adaptive_grid_MCapprox(epsilon, N, SafeSet);
            break;
          }
          case 222: {
            std::cout << "This options is not available yet. Work in progress.";
            taskFAUST.Adaptive_grid_ReachAvoidMCapprox(epsilon, N, SafeSet,
                                                       TargetSet);
            break;
          }
          default: {
            std::cout << "This options is not available yet. Work in progress.";
            exit(0);
            break;
          }
          }
          std::cout << "The abstraction consists of " << taskFAUST.X.n_rows
                    << " representative points." << std::endl;
          if (taskFAUST.X.n_rows > 1000000) {
            std::cout << "Abstraction is too large, need more memory"
                      << std::endl;
            exit(0);
          }
          // Because of taking the center points as representative points the
          // resulting error is half of the outcome error.
          taskFAUST.E = 0.5 * taskFAUST.E;

          // Creation of Markov Chain
          if (Distribution == 2) {
            taskFAUST.MCapprox(epsilon);
          } else {
            taskFAUST.MCcreator(epsilon);
          }
          // Calculation of the resulting problem
          if (input.myModel.Q == 1) {
            switch (Problem) {
            case 100: {
              taskFAUST.StandardProbSafety(N);
              end = clock();
              time = (double)(end - begin) / CLOCKS_PER_SEC;
              break;
            }
            case 200: {
              taskFAUST.StandardReachAvoid(TargetSet, N);
              end = clock();
              time = (double)(end - begin) / CLOCKS_PER_SEC;
              break;
            }
            default: {
              std::cout
                  << "This options is not available yet. Work in progress.";
              exit(0);
            } break;
            }
          } else {
            Tp_all.push_back(taskFAUST.Tp);
          }
          //  }
          break;
        }
        case 1: {
          switch (task2Solve) {
          case 111: {
            taskFAUST.Uniform_grid_Contr(epsilon, N, SafeSet, InputSet);
            break;
          }
          case 121: {
            taskFAUST.Uniform_grid_Contr(10 * epsilon, N, SafeSet, InputSet);
            taskFAUST.Adaptive_grid_multicell_Contr(epsilon, N, SafeSet,
                                                    InputSet);
            break;
          }
          case 131: {
            taskFAUST.Uniform_grid_Contr(10 * epsilon, N, SafeSet, InputSet);
            taskFAUST.Adaptive_grid_semilocal_Contr(epsilon, N, SafeSet,
                                                    InputSet);
            break;
          }
          case 211: {
            taskFAUST.Uniform_grid_ReachAvoid_Contr(epsilon, N, SafeSet,
                                                    InputSet, TargetSet);
            break;
          }
          case 221: {
            taskFAUST.Uniform_grid_ReachAvoid_Contr(10 * epsilon, N, SafeSet,
                                                    InputSet, TargetSet);
            taskFAUST.Adaptive_grid_ReachAvoid(epsilon, N, SafeSet, TargetSet);
            break;
          }
          case 231: {
            std::cout << "This options is not available yet. Work in progress.";
            exit(0);
            taskFAUST.Uniform_grid_ReachAvoid_Contr(10 * epsilon, N, SafeSet,
                                                    InputSet, TargetSet);
            //    taskFAUST.Adaptive_grid_ReachAvoid_semilocal(epsilon,N,SafeSet,TargetSet);
            break;
          }
          case 112: {
            taskFAUST.Uniform_grid_MCapprox_Contr(epsilon, N, SafeSet,
                                                  InputSet);
            break;
          }
          case 212: {
            taskFAUST.Uniform_grid_ReachAvoidMCapprox_Contr(
                epsilon, N, SafeSet, InputSet, TargetSet);
            break;
          }
          case 222: {
            taskFAUST.Adaptive_grid_ReachAvoidMCapprox(epsilon, N, SafeSet,
                                                       TargetSet);
            break;
          }
          default: {
            std::cout << "This options is not available yet. Work in progress.";
            exit(0);
            break;
          }
          }
          std::cout << "The abstraction consists of " << taskFAUST.X.n_rows
                    << " state representative points." << std::endl;
          std::cout << "The abstraction consists of " << taskFAUST.U.n_rows
                    << " input representative points." << std::endl;
          if (taskFAUST.X.n_rows * taskFAUST.U.n_rows > 100000) {
            std::cout << "Abstraction is too large, need more memory"
                      << std::endl;
            exit(0);
          }
          // Because of taking the center points as representative points the
          // resulting error is half of the outcome error.
          taskFAUST.E = 0.5 * taskFAUST.E;
          // Creation of Markov Chain
          if (Distribution == 2) {
            taskFAUST.MCapprox_Contr(epsilon, model);
          } else {
            taskFAUST.MCcreator_Contr(epsilon);
          }
          // Calculation of the resulting problem
          if (input.myModel.Q == 1) {
            switch (Problem) {
            case 100: {
              taskFAUST.StandardProbSafety_Contr(N);
              end = clock();
              time = (double)(end - begin) / CLOCKS_PER_SEC;
              break;
            }
            case 200: {
              taskFAUST.StandardReachAvoid_Contr(TargetSet, N);
              end = clock();
              time = (double)(end - begin) / CLOCKS_PER_SEC;
              break;
            }
            default: {
              std::cout
                  << "This options is not available yet. Work in progress.";
              exit(0);
            } break;
            }
          } else {
            Tp_all.push_back(taskFAUST.Tp);
          }
        }
        }
        // Get current time to time stamp outputs
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
        auto str = oss.str();

        // Rescale grid axis to original axis
        arma::vec inter_d = arma::ones<arma::vec>(taskFAUST.X.n_rows);
        for (unsigned d = 0; d < taskFAUST.X.n_cols / 2; d++) {
          inter_d = j(d, d) * inter_d;
          taskFAUST.X.col(d) = inter_d % taskFAUST.X.col(d);
          inter_d = arma::ones<arma::vec>(taskFAUST.X.n_rows);
        }
        if (input.myModel.Q == 1) {
          taskFAUST.formatOutput(time, str, Problem, N);
        }
      }
      // For multiple modes only now that I have obtained all the
      //  Transition probabilities I need to perform parallel composition
      // and compute overall MDP
      // Tp_all contains all the Tp matrices
      // For Safety ---------------------------------------
      // TODO Check > 2D for safety + REACH AVOID
      if (input.myModel.Q > 1) {
        arma::mat Tq = input.myModel.Tq;
        // for each mode compute V
        // first get Tp for first control
        std::vector<arma::mat> Tpall;
        for (size_t qd = 0; qd < input.myModel.Q; qd++) {
          Tpall.push_back(Tp_all[qd][0]);
        }
        // Compute V for first one
        taskFAUST.StandardProbSafety(input.myModel.Q, Tq, Tpall, N);
        // Where we are storing all the Value functions for each control
        int u_dim = 1;
        if (!taskFAUST.U.is_empty()) {
          u_dim = taskFAUST.U.n_rows;
        }
        arma::mat Vall(taskFAUST.V.n_rows, u_dim);
        // store first computed V for first control in first column
        Vall.col(0) = taskFAUST.V;

        // Iteratively compute for all remaining control
        Tpall.clear();
        for (size_t ud = 1; ud < u_dim; ud++) {
          for (size_t qd = 0; qd < input.myModel.Q; qd++) {
            Tpall.push_back(Tp_all[qd][ud]);
          }
          // Compute the next value function
          taskFAUST.StandardProbSafety(input.myModel.Q, Tq, Tpall, N);
          // Store result
          Vall.col(ud) = taskFAUST.V;
        }
        // Now to get final value  and optimal policy need to find
        // V with max and corresponding column index
        arma::mat newV = arma::max(Vall, 1);
        taskFAUST.V = newV;

        arma::ucolvec index = arma::index_max(Vall, 1);
        taskFAUST.OptimalPol = index;

        //-------------------------------------------------------
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
        auto str = oss.str();
        taskFAUST.formatOutput(time, str, Problem, N);
      }
    } catch (const std::bad_alloc &e) {
      std::cerr << e.what() << std::endl;
      exit(0);
    }
    break;
  }
  case 3: // model checking using BMDP
  {
    // Initialise timers
    clock_t begin, end;
    double time;
    begin = clock();

    // Get model definitions
    input.myModel.x_mod[0].F = arma::eye(input.myModel.n, input.myModel.n);
    bmdp_t taskBMDP(input.myTask, input.myModel);

    // Check input model
    int num_cont = input.myModel.n;
    for (int i = 0; i < input.myModel.Q; ++i) {
      if (input.myModel.x_mod[i].A.n_rows != num_cont) {
        std::cout << "Different number of continuous variables for each mode, "
                     "currently not supported. Work in progress"
                  << std::endl;
        exit(0);
      }
      if (input.myModel.x_mod[i].sigma.n_rows != num_cont) {
        std::cout << "Different number of continuous variables for each mode, "
                     "currently not supported. Work in progress"
                  << std::endl;
        exit(0);
      }
    }
    //  if (input.myModel.Q > 2) {
    //    std::cout << "Work in progress, will be available shortly" <<
    //    std::endl; exit(0);
    //  }
    // Set number of actions
    taskBMDP.actNum = input.myModel.Q;

    for (int i = 0; i < input.myModel.Q; ++i) {
      if (i == 0) {
        arma::mat j = arma::zeros<arma::mat>(1, input.myModel.n);
        taskBMDP.desc.mean[0] = j;
        taskBMDP.desc.cov[0] = taskBMDP.desc.dyn.dynamics[i].sigma;
      } else {
        arma::mat j = arma::zeros<arma::mat>(1, input.myModel.n);
        taskBMDP.desc.mean.push_back(j);
        taskBMDP.desc.cov.push_back(taskBMDP.desc.dyn.dynamics[i].sigma);
      }
    }
    taskBMDP.desc.boundary = input.myTask.boundary;
    taskBMDP.desc.gridsize = input.myTask.gridsize;
    if (taskBMDP.desc.boundary.n_rows != taskBMDP.desc.gridsize.n_cols) {
      std::cout << "Incorrect boundary or grid size" << std::endl;
      exit(0);
    }
    taskBMDP.desc.reftol = input.myTask.reftol;
    if (taskBMDP.desc.reftol.n_cols != taskBMDP.desc.gridsize.n_cols) {
      std::cout << "Incorrect relative tolerance parameter input" << std::endl;
      exit(0);
    }
    // Check if need to rescale boundary
    arma::mat proj = arma::eye<arma::mat>(taskBMDP.desc.boundary.n_rows,
                                          taskBMDP.desc.boundary.n_rows);
    /*  arma::mat max_Ss = arma::max(taskBMDP.desc.boundary );
      arma::mat min_Ss = arma::min(taskBMDP.desc.boundary );
      double diff_Ss = max_Ss(1) - min_Ss(0);
      if(diff_Ss > 100) {
        // Identify row with smallest values to rescale to that
        arma::mat min_row = arma::min(taskBMDP.desc.boundary ,0);
        arma::mat OrSS = taskBMDP.desc.boundary ; // The original Safeset
      definition arma::mat to_inv = OrSS; for(unsigned i =0; i < OrSS.n_rows;
      i++) { taskBMDP.desc.boundary.row(i) = min_row;
        }
        arma::mat y_inv = arma::diagmat(min_row);
        arma::mat j_bar = OrSS/y_inv;
        proj = j_bar.replace(arma::datum::inf, 0);
        for(unsigned k = 0; k < input.myModel.Q; k++) {
          taskBMDP.desc.dyn.dynamics[k].sigma =
      arma::inv(proj)*taskBMDP.desc.dyn.dynamics[k].sigma;
        }
      }*/

    taskBMDP.eps = input.myTask.eps;
    // Constructing the abstraction
    std::cout << "Constructing the IMDP abstraction " << std::endl;

    // Identify if performing synthesis or verification
    // and whether safety or reach avoid
    int RA = 0;
    if (input.myTask.propertySpec == 2 || input.myTask.propertySpec == 4) {
      RA = 1;
    }
    taskBMDP.bmdpAbstraction(input.myTask.T, RA);
    std::cout << "Done with abstraction construction" << std::endl;
    if (input.myTask.propertySpec == 1 || input.myTask.propertySpec == 3) {
      arma::uvec phi1 = arma::ones<arma::uvec>(taskBMDP.Stepsmax.n_cols);
      phi1(phi1.n_rows - 1, 0) = 0;
      arma::uvec labels = arma::zeros<arma::uvec>(taskBMDP.Stepsmax.n_cols);
      labels(labels.n_rows - 1, 0) = 1;
      std::cout << "Performing  model checking " << std::endl;
      taskBMDP.createSynthFile(phi1, labels);
      if (input.myTask.propertySpec == 1) {
        taskBMDP.runSafety(1e-4, input.myTask.T);
      } else {
        taskBMDP.runSynthesis(1e-4, input.myTask.T);
      }
    } else {
      // Get coordinates of labels for phi1 and phi 2
      arma::uvec phi1 =
          taskBMDP.getLabels("../phi1.txt", input.myModel.n, true);

      // Since not phi1 need to negate phi1
      for (unsigned i = 0; i < phi1.n_rows; i++) {
        if (phi1(i) == 1) {
          phi1(i) = 0;
        } else {
          phi1(i) = 1;
        }
      }
      arma::uvec phi2 =
          taskBMDP.getLabels("../phi2.txt", input.myModel.n, false);
      std::cout << "Performing  model checking " << std::endl;
      taskBMDP.createSynthFile(phi1, phi2);

      if (input.myTask.propertySpec == 2) {
        taskBMDP.runSafety(1e-4, input.myTask.T);
      } else {
        taskBMDP.runSynthesis(1e-4, input.myTask.T);
      }
    }

    end = clock();
    time = (double)(end - begin) / CLOCKS_PER_SEC;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ofstream myfile;
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    auto str = oss.str();

    // TODO: Rescale grid axis to original axis
    int dim = taskBMDP.desc.boundary.n_rows;
    if (dim == 2) {
      arma::mat inter_d = arma::ones<arma::mat>(1, dim);
      for (unsigned i = 0; i < taskBMDP.mode.size(); ++i) {
        for (unsigned p = 0; p < taskBMDP.mode[i].vertices.size(); p++) {
          for (unsigned d = 0; d < dim; d++) {
            inter_d = proj(d, d) * inter_d;
            if (taskBMDP.mode[i].vertices[p].n_cols == 4) {
              taskBMDP.mode[i].vertices[p].resize(dim, 2);
            }
            taskBMDP.mode[i].vertices[p].row(d) =
                inter_d % taskBMDP.mode[i].vertices[p].row(d);
            inter_d = arma::ones<arma::mat>(1, dim);
          }
        }
      }
      for (unsigned d = 0; d < dim; d++) {
        inter_d = proj(d, d) * inter_d;
        if (taskBMDP.desc.boundary.n_cols == 4) {
          arma::vec minb = arma::min(taskBMDP.desc.boundary, 1);
          arma::vec maxb = arma::max(taskBMDP.desc.boundary, 1);
          taskBMDP.desc.boundary.resize(dim, 2);
          taskBMDP.desc.boundary.col(0) = minb;
          taskBMDP.desc.boundary.col(1) = maxb;
        }
        taskBMDP.desc.boundary.row(d) = inter_d % taskBMDP.desc.boundary.row(d);
        inter_d = arma::ones<arma::mat>(1, dim);
      }
    }
    taskBMDP.formatOutput(time, str);

    // Simulation of BMDP
    if (input.myTask.propertySpec == 3 && input.myModel.n) {
      std::string exportOpt;
      std::string str("y");
      std::string str1("yes");
      std::string str2("YES");
      std::cout << "Would you like to simulate evolution under synthesised "
                   "policy [y- yes, n - no]"
                << std::endl;
      std::cin >> exportOpt;
      if ((exportOpt.compare(str) == 0) || (exportOpt.compare(str1) == 0) ||
          (exportOpt.compare(str2) == 0)) {
        std::cout << "Simulation of model using optimal policy" << std::endl;
        arma::mat init(2, 1);
        init(0, 0) = -0.5;
        init(1, 0) = -1;
        int mode = 1;
        arma::mat policy = taskBMDP.Policy;

        // Find IMDP state you are in; we start from mode 1
        int init_state = 0;
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev());
        std::normal_distribution<double> d2{0, 1};
        for (unsigned i = 0; i < taskBMDP.mode[0].vertices.size(); i++) {
          arma::mat crnt_vertex = taskBMDP.mode[0].vertices[i];
          int inpoly = pnpoly(1, crnt_vertex.row(0), crnt_vertex.row(1),
                              init(0, 0), init(1, 0));
          if (inpoly) {
            init_state = i;
            break;
          }
        }

        arma::mat prev(2, 10);
        prev.col(0) = init;
        arma::mat W(2, 1);
        for (unsigned t = 0; t < 9; t++) {
          for (unsigned j = 0; j < 2; j++) {
            W(j, 0) = d2(generator);
          }
          arma::mat x_new =
              input.myModel.x_mod[policy(init_state, 0)].A * prev.col(t) +
              input.myModel.x_mod[policy(init_state, 0)].sigma * W;
          if (x_new(0, 0) > arma::max(taskBMDP.desc.boundary.row(0))) {
            x_new(0, 0) = arma::max(taskBMDP.desc.boundary.row(0));
          }
          if (x_new(1, 0) > arma::max(taskBMDP.desc.boundary.row(1))) {
            x_new(1, 0) = arma::max(taskBMDP.desc.boundary.row(1));
          }
          if (x_new(1, 0) < arma::min(taskBMDP.desc.boundary.row(1))) {
            x_new(1, 0) = arma::min(taskBMDP.desc.boundary.row(1));
          }
          if (x_new(0, 0) < arma::min(taskBMDP.desc.boundary.row(0))) {
            x_new(0, 0) = arma::min(taskBMDP.desc.boundary.row(0));
          }
          for (unsigned i = 0;
               i < taskBMDP.mode[policy(init_state, 0)].vertices.size(); i++) {
            arma::mat crnt_vertex =
                taskBMDP.mode[policy(init_state, 0)].vertices[i];
            int inpoly = pnpoly(1, crnt_vertex.row(0), crnt_vertex.row(1),
                                x_new(0, 0), x_new(1, 0));
            if (inpoly) {
              init_state = i;
              break;
            }
          }
          prev.col(t + 1) = x_new;
        }
        std::string sim_name = "../results/Simulation_" + str + ".txt";
        prev = prev.t();
        prev.save(sim_name, arma::raw_ascii);
      } else {
        std::cout << "Done!" << std::endl;
      }
    }
    break;
  }
  default: {
    std::cout << "No task specification given" << std::endl;
    exit(0);
  }
  }
}

#endif
