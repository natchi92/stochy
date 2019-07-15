/*
 * Stochy main file
 *
 *  Created on: 14 Nov 2017
 *      Author: nathalie
 */

#include "taskExec.h"
#include "time.h"
#include <iostream>

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

  // ------------------------- Case study 2 - Strategy synthesis
  // -----------------------------------
  std::cout << "------------ Performing Case Study 2 : Strategy synthesis  "
               "-----------"
            << std::endl;
  std::cout << std::endl;


  // Define Time horizon
  // Infinite time horizon:  T = -1
  // Finite time horizon: T = k (where k is an integer value)
  int T = -1;

  // Define grid boundary
  // [x-coordinates; y coordinates]
  arma::mat bound = {{-1.5, 1.5}, {-1.5, 1.5}};

  // Define grid size for each dimension
  arma::mat grid = {{0.12, 0.12}};

  // Define tolerance
  arma::mat reft = {{0.06, 0.06}};

  // Define dynamics of stochastic process
  // Stochastic process is described for each mode
  // using the kernel Tx = \N(Aqix, Gqi) where
  // \N is the normal distribution with mean and variance
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

  // Task definition (1 = simulator, 2 = faust^2, 3 = imdp)
  int lb  =imdp;

  // Property type
  // (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
  int p = synthesis_reach_avoid;

  // Specify task to be performed
  taskSpec_t cs2Spec(lb, T, p, bound, grid, reft);

  // Combin specfication and associated task
  inputSpec_t<arma::mat, int> cs2Input(cs2SHS, cs2Spec);

  // Perform Task
  performTask(cs2Input);

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "------------Completed Case Study 2 : Results in results "
               "folder -----------"
            << std::endl;
  std::cout << std::endl;
}
