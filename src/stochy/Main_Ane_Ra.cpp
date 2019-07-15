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

  // ------------------------- Case study 1 - Verification
  // -----------------------------------
  std::cout << "------------ Performing Anaesthesia model  :  Problem 2.1.1 "
               "of ARCH -----------"
            << std::endl;
  std::cout << std::endl;
  // --------------------------------------------
  arma::mat Aq0 = {{0.8192, 0.0341, 0.0126}, {0.0165, 0.9822, 0.0001},{0.0009, 1e-4, 0.9989}};
  arma::vec Bq0 = {{0.1105},{0.0012},{0.0001}};
  arma::mat Gq0 = { {1.1905, 0, 0}, {0 , 1.1905,0 }, {0, 0, 1.905}};
  arma::vec Qq0 = {{0}, {0}, {0}};
  ssmodels_t model(Aq0,Bq0,Qq0,Gq0);

  std::vector<ssmodels_t> models1 = {model};
  shs_t<arma::mat,int> cs1SHS(models1);

  // Define max error
  double eps = .18;

  // Define safe set
  arma::mat Safe = {{0.2381, 1.4286}, {0, 2.381}, {0, 2.381}};

  arma::mat Input = {{0, 1}};

  arma::mat Target = { {0.9523, 1.428}, {1.904, 2.381},{1.904, 2.381}};

  // Define grid type
  // (1 = uniform, 2 = adaptive)
  // For comparison with IMDP we use uniform grid
  int gridType = 1;

  // Time horizon
  int K = 1;

  // Library (1 = simulator, 2 = faust^2, 3 = imdp)
  int lb = 2;

  // Property type
  // FIX (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
  int p = 2;

  // Task specification
  //taskSpec_t cs1SpecFAUST(lb, K, p, Safe, Input, eps,gridType,1);
  taskSpec_t cs1SpecFAUST(lb, K, p, Safe, Target, Input, eps, gridType, 1);

  // Combine model and associated task
  inputSpec_t<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

  // Perform  Task
  performTask(cs1InputFAUST);


  // Perform  Task
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "------------Completed -----------"
            << std::endl;
}
