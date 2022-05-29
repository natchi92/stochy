/*
 * Stochy main file
 *
 *  Created on: 14 Nov 2017
 *      Author: nathalie
 */

#include "taskExec.h"
#include "time.h"
#include <iostream>
#include <nlopt.hpp>

int main(int argc, char **argv) {

  std::cout << " _______  _______  _______  _______  __   __  __   __ "
            << std::endl;
  std::cout << "|       ||       ||       ||       ||  | |  ||  | |  |"
            << std::endl;
  std::cout << "|  _____||_     _||   _   ||       ||  |_|  ||  |_|  |"
            << std::endl;
  std::cout << "| |_____   |   |  |  | |  ||       ||       ||       |"
            << std::endl;
  std::cout << "|_____  |  |   |  |  |_|  ||      _||       ||_     _|"
            << std::endl;
  std::cout << " _____| |  |   |  |       ||     |_ |   _   |  |   |  "
            << std::endl;
  std::cout << "|_______|  |___|  |_______||_______||__| |__|  |___| "
            << std::endl;
  std::cout << std::endl;
  std::cout << " Welcome!  Copyright (C) 2018  natchi92 " << std::endl;
  std::cout << std::endl;

  // Input case study number
  // Case study 1 : Verification
  // Case study 2 : Synthesis
  // Case study 3 : Scaling in dimensions
  // Case study 4 : Simulation
  if (argc < 2) {
    std::cout
        << "No case study selection was given, please make use of ./stochy i, "
           "where i=[1,2,3,4] is the required case study number"
        << std::endl;
    exit(0);
  }
  int selection = strtol(argv[1], NULL, 0);
  switch (selection) {
  case 1: {
    // ------------------------- Case study 1 - Verification
    // -----------------------------------
    std::cout << "------------ Performing Case Study 1 : Formal Verification "
                 "of CO2 model -----------"
              << std::endl;
    std::cout << std::endl;

    // Get model from file

    shs_t<arma::mat, int> cs1SHS("CS1.mat");

    // Specify verification task
    std::vector<int> grid_size;
    int num = -1;
    do {
      std::cout << "Please input grid sizes of interest [value, value, ..], "
                   "enter -1 to generate results for all grids in Table 1 of "
                   "paper or 0 to stop: ";
      std::cin >> num;
      if (num != 0) {
        grid_size.push_back(num);
      }
    } while (num > 0);
    if (grid_size.empty()) {
      std::cout << "No dimension provided." << std::endl;
      exit(0);
    }

    // Get number of grid to generate
    // For FAUST set the error value
    // For IMDP set the gridsize
    std::vector<double> E_FAUST;
    std::vector<double> GS_IMDP;
    if (num == -1) {
      std::vector<double> F = {2.5, 1.8, 1.25, 1.02, 0.90};
      E_FAUST = F;
      std::vector<double> G = {0.25, 0.1818, 0.125, 0.1017, 0.09231};
      GS_IMDP = G;
      std::vector<int> states = {576, 1089, 2304, 3481, 4225};
      grid_size = states;
    } else {
      for (unsigned i = 0; i < grid_size.size(); i++) {
        switch (grid_size[i]) {
        case 576: {
          E_FAUST.push_back(2.5);
          GS_IMDP.push_back(0.25);
          break;
        }
        case 1089: {
          E_FAUST.push_back(1.8);
          GS_IMDP.push_back(0.1818);
          break;
        }
        case 2304: {
          E_FAUST.push_back(1.25);
          GS_IMDP.push_back(0.125);
          break;
        }
        case 3481: {
          E_FAUST.push_back(1.02);
          GS_IMDP.push_back(0.1017);
          break;
        }
        case 4225: {
          E_FAUST.push_back(.90);
          GS_IMDP.push_back(0.09231);
          break;
        }
        default: {
          std::cout << "Invalid grid size" << std::endl;
          int E_F = 0, G_I = 0;
          std::cout << " Input FAUST max error required: ";
          std::cin >> E_F;
          E_FAUST.push_back(E_F);
          std::cout << std::endl;
          std::cout << " Input IMDP grid size : ";
          std::cin >> G_I;
          if (G_I <= 0) {
            std::cout << "Invalid grid size. Grid size cannot be equal to 0 or "
                         "negative"
                      << std::endl;
            exit(0);
          }
          GS_IMDP.push_back(G_I);
        }
        }
      }
    }

    for (unsigned g = 0; g < grid_size.size(); g++) {
      // get number of states
      int num_states = grid_size[g];
      std::cout << " Formal verification via FAUST^2 with grid of "
                << num_states << " states" << std::endl;
      // Define max error
      double eps = E_FAUST[g];

      // Define safe set
      arma::mat Safe = {{18, 24}, {18, 24}};

      // Define grid type
      int gridType = 2; // Grid: 1 = uniform, 2 = adaptive

      // Time horizon
      int K = 3;
      faust_t inputFAUST;
      taskSpec_t cs1SpecFAUST(2, K, 1, Safe, eps, gridType);

      // Combine model and associated task
      inputSpec_t<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

      // Perform  Task
      performTask(cs1InputFAUST);

      // Perform  Task
      std::cout << " Done with with FAUST^2 with" << num_states << " states "
                << std::endl;
      std::cout << "----------------------------------------" << std::endl;
      std::cout << " Formal verification via IMDP with grid of " << num_states
                << " states" << std::endl;
      //  Define safe set
      arma::mat bound = {{18, 24}, {18, 24}};
      // Define grid size for 576 states
      arma::mat grid1 = {{GS_IMDP[g], GS_IMDP[g]}};

      // Relative tolerance
      arma::mat rtol1 = {{1, 1}};

      // Specify task to be performed
      taskSpec_t cs1Spec(3, K, 1, bound, grid1, rtol1);

      // Combine model and associated task
      cs1SHS.x_mod[0].F = arma::eye(2, 2);
      inputSpec_t<arma::mat, int> cs1Input(cs1SHS, cs1Spec);

      // Perform Task
      performTask(cs1Input);

      std::cout << "------------Completed Case Study 1 : Results in results "
                   "folder -----------"
                << std::endl;
    }
    break;
  }
  case 2: {
    // ------------------------- Case study 2 - Strategy synthesis
    // -----------------------------------
    std::cout << "------------ Performing Case Study 2 : Strategy synthesis  "
                 "-----------"
              << std::endl;
    std::cout << std::endl;

    // Define Time horizon
    int T = -1;

    // Define grid boundary
    // [x-coordinates; y coordinates]
    arma::mat boundary = {{-1.5, 1.5}, {-1.5, 1.5}};

    // Define grid size for each dimension
    arma::mat grid = {{0.12, 0.12}};

    // Define tolerance
    arma::mat rtol = {{0.06, 0.06}};

    // Define dynamics of stochastic process
    // Mode q1
    arma::mat Aq1 = {{0.43, 0.52}, {0.65, 0.12}};
    arma::mat Gq1 = {{1, 0.1}, {0, 0.1}};
    ssmodels_t modelq1(Aq1, Gq1);

    // Mode q2
    arma::mat Aq2 = {{0.65, 0.12}, {0.52, 0.43}};
    arma::mat Gq2 = {{0.2, 0}, {0, 0.2}};
    ssmodels_t modelq2(Aq2, Gq2);

    // Combine model to shs
    std::vector<ssmodels_t> models = {modelq1, modelq2};
    int x_dim = 2;
    shs_t<arma::mat, int> cs2SHS(x_dim, models);

    // Specify task to be performed
    taskSpec_t cs2Spec(3, T, 3, boundary, grid, rtol);

    // Comspecfication and associated task
    inputSpec_t<arma::mat, int> cs2Input(cs2SHS, cs2Spec);

    // Perform Task
    performTask(cs2Input);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "------------Completed Case Study 2 : Results in results "
                 "folder -----------"
              << std::endl;
    std::cout << std::endl;
    break;
  }
  case 3: {
    // ------------------------- Case study 3 - Scaling in dimensions
    // -----------------------------------
    std::cout << "------------ Performing Case Study 3 : Scaling in number of "
                 "dimensions -----------"
              << std::endl;
    std::cout << std::endl;

    // Specify number of dimensions of interest
    std::vector<int> dim;
    int num = -1;
    do {
      std::cout << "Please input dimension of models of interest  [value, "
                   "value , ..], enter -1 to generate results for all "
                   "dimensions in Table 2 of paper or 0 to stop: ";
      std::cin >> num;
      if (num != 0) {
        dim.push_back(num);
      }
    } while (num > 0);

    if (dim.empty()) {
      std::cout << "No dimension provided." << std::endl;
      exit(0);
    }
    // Get dimensions to run
    std::vector<int> dimensions_2Run;
    if (num == -1) {
      std::vector<int> F = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
      dimensions_2Run = F;
    } else {
      dimensions_2Run = dim;
    }

    // ------------------------- Case study 3 - Scaling in dimensions
    // -----------------------
    for (unsigned d = 0; d < dimensions_2Run.size(); d++) {
      std::cout
          << "-----------------------------------------------------------------"
          << std::endl;
      std::cout << std::endl;
      // Get current dimension
      int crnt_dim = dimensions_2Run[d];
      std::cout << "Running verification with " << crnt_dim
                << " continuous variables." << std::endl;

      // Define the boundary for each dimesnion
      arma::mat bound(1, 2);
      bound(0, 0) = -1;
      bound(0, 1) = 1;

      // Define grid size for each dimension
      arma::mat grid = arma::ones<arma::mat>(1, 1);

      // Define relative tolerance for each dimension
      arma::mat reft = arma::ones<arma::mat>(1, 1);

      // Concatenate matrices to correct dimension
      for (unsigned con = 0; con < crnt_dim - 1; con++) {
        arma::mat tempBound(1, 2);
        tempBound(0, 0) = -1;
        tempBound(0, 1) = 1;
        bound = join_vert(bound, tempBound);
        grid = join_horiz(grid, arma::ones<arma::mat>(1, 1));
        reft = join_horiz(reft, arma::ones<arma::mat>(1, 1));
      }
      // Define time horizon
      int T = -1;

      // Select task to perform
      // 1 = verification, 2 = synthe:sis for propertySpec
      taskSpec_t cs3Spec(3, T, 1, bound, grid, reft);

      // set appropriate dynamics
      arma::mat Am3 = 0.8 * arma::eye(crnt_dim, crnt_dim);
      arma::mat sigmam3 = 0.2 * arma::eye(crnt_dim, crnt_dim);

      ssmodels_t model03(Am3, sigmam3);

      std::vector<ssmodels_t> models4 = {model03};
      shs_t<arma::mat, int> cs3SHS(crnt_dim, models4);

      inputSpec_t<arma::mat, int> cs3Input(cs3SHS, cs3Spec);

      performTask(cs3Input);
      std::cout
          << "-----------------------------------------------------------------"
          << std::endl;
      std::cout << "Completed verfication task." << std::endl;
      std::cout
          << "-----------------------------------------------------------------"
          << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "------------Completed Case Study 3 : Results in results "
                 "folder -----------"
              << std::endl;
    break;
  }
  case 4: {
    // ------------------------- Case study 4 - Simulation
    // -----------------------------------
    std::cout << "------------ Performing Case Study 4 : Simulation of CO2 "
                 "model  -----------"
              << std::endl;
    std::cout << std::endl;

    // Define time horizon for simulation
    int T1 = 32;

    // Number of monte carlo simulations
    int monte = 5000;

    // Define initial distributions
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());

    arma::mat Xinit = join_vert(arma::ones<arma::mat>(1, monte),
                                15 * arma::ones<arma::mat>(1, monte));
    std::normal_distribution<double> d1{450, 25};
    std::normal_distribution<double> d2{17, 2};
    for (size_t i = 0; i < monte; ++i) {
      Xinit(0, i) = d1(generator);
      Xinit(1, i) = d2(generator);
    }

    std::vector<arma::mat> X = {Xinit};
    std::vector<arma::mat> q_init = {arma::zeros<arma::mat>(T1 + 1, monte)};

    // Read input signal from file
    arma::mat U = readInputSignal("u.txt");
    ExData data(X, U, q_init);

    // Get model from file
    shs_t<arma::mat, int> cs4SHS("CS4.mat", data);

    // Specify task to be performed
    taskSpec_t cs4Spec(1, T1, monte);

    // Combine model and associated task
    inputSpec_t<arma::mat, int> cs4Input(cs4SHS, cs4Spec);

    // Perform Task
    performTask(cs4Input);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "------------Completed Case Study 4 : Results in results "
                 "folder -----------"
              << std::endl;
    break;
  }
  default: {
    std::cout << " Invalid case study selection " << std::endl;
    break;
  }
  }
}
