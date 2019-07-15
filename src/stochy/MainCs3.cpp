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

  // ------------------------- Case study 3 - Scaling in dimensions
  // -----------------------------------
  std::cout << "------------ Performing Case Study 3 : Scaling in number of "
               "dimensions -----------"
            << std::endl;
  std::cout << std::endl;

  // Get current dimension
  int d = 2;
  std::cout << "Running verification with " << d
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
  // Boundary = [-1, 1]^d
  // Grid = [1]^d
  // Reft = [1]^d
  for (unsigned con = 0; con < d- 1; con++) {
    arma::mat tempBound(1, 2);
    tempBound(0, 0) = -1;
    tempBound(0, 1) = 1;
    bound = join_vert(bound, tempBound);
    grid = join_horiz(grid, arma::ones<arma::mat>(1, 1));
    reft = join_horiz(reft, arma::ones<arma::mat>(1, 1));
  }
  // Define time horizon
  // Infinite time horizon:  T = -1
  // Finite time horizon: T = k (where k is an integer value)
  int T = -1;

  // Task definition (1 = simulator, 2 = faust^2, 3 = imdp)
  int lb  =imdp;

  // Property type
  // (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
  int p = verify_safety;

  // task specification
  taskSpec_t cs3Spec(lb, T, p, bound, grid, reft);

  // Define model dynamics
  arma::mat Am3 = 0.8 * arma::eye(d,d);
  arma::mat Gm3 = 0.2 * arma::eye(d,d);

  ssmodels_t model03(Am3, Gm3);

  std::vector<ssmodels_t> models4 = {model03};
  shs_t<arma::mat, int> cs3SHS(d, models4);

  // Combine
  inputSpec_t<arma::mat, int> cs3Input(cs3SHS, cs3Spec);

  // Perform task
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
