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

  // ------------------------- Case study 1 - Verification
  // -----------------------------------
  std::cout << "------------ Performing Case Study 1 : Formal Verification "
               "of CO2 model -----------"
            << std::endl;
  std::cout << std::endl;

  // Get model from file
  shs_t<arma::mat, int> cs1SHS("CS1.mat");

  // To specify model manually
  // comment the previous line of code
  // i.e. shs_t<arma::mat,int> cs1SHS("CS1.mat'")
  // and then uncomment the following lines of code
  // (i.e. lines 50-55) which allow you to manually
  // define the model dynamics
  // ---------------------------------------------
   /*arma::mat Aq0 = {{0.935, 0},{0, 0.9157}};
   arma::vec Fq0 = arma::zeros<arma::mat>(2,1);
   arma::mat Gq0 = {{1.7820, 0},{0, 0.5110}};
   ssmodels_t model(Aq0,Fq0,Gq0);
   std::vector<ssmodels_t> models = {model};
   shs_t<arma::mat,int> cs1SHS(models);*/
  // --------------------------------------------

  // Specify verification task
  // Starting with MDP method
  std::cout << "Formal verification via MDP" << std::endl;
  std::cout << std::endl;
  // Define max error
  // Maximal error to obtain specific state space sizes
  // --------------------------------------------------------
  // State space size| 576 | 1089 | 2304  |  3481 | 4225
  // ________________________________________________________
  // Max Error       | 2.5 | 1.8  | 1.25  | 1.02  | 0.90
  // --------------------------------------------------------

  // Define max error for 576 states
  double eps = 2.5;

  // Define safe set
  arma::mat Safe = {{405, 540}, {18, 24}};

  // Define grid type
  // (1 = uniform, 2 = adaptive)
  // For comparison with IMDP we use uniform grid
  int gridType = uniform;

  // Time horizon
  int K = 3;

  // Property type
  int p = verify_safety;
  // Library
  int lb = mdp;


  // Task specification
  taskSpec_t cs1SpecFAUST(lb, K, p, Safe, eps, gridType);

  // Combine model and associated task
  inputSpec_t<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

  // Perform  Task
  performTask(cs1InputFAUST);

  // Perform  Task
  std::cout << " Done with with FAUST^2 "   << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << " Formal verification via IMDP" << std::endl;

  //  Define safe set
  arma::mat bound = {{405, 540}, {18, 24}};

  // Grid size to obtain specific state space sizes
  // --------------------------------------------------------
  // State space size| 576 | 1089   | 2304  |  3481 | 4225
  // ________________________________________________________
  // Grid size       | 0.25 | 0.1818 | 0.125 | 0.1017| 0.09231
  // --------------------------------------------------------

  // Define grid size for 576 states
  arma::mat grid1 = {{0.25, 0.25}};

  // Relative tolerance
  arma::mat rtol1 = {{1, 1}};

  // Library (1 = simulator, 2 = faust^2, 3 = imdp)
  lb = imdp;

  // Specify task to be performed
  taskSpec_t cs1Spec(lb, K, p, bound, grid1, rtol1);

  // Combine model and associated task
  inputSpec_t<arma::mat, int> cs1Input(cs1SHS, cs1Spec);

  // Perform Task
  performTask(cs1Input);

  std::cout << "------------Completed Case Study 1 : Results in results "
               "folder -----------"
            << std::endl;
}
