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
  std::cout << "------------ Performing BAS model  :  CS1BASssa"
               "of ARCH -----------"
            << std::endl;
  std::cout << std::endl;
  // --------------------------------------------

  //arma::mat Aq0 = { {0.6682, 0, 0.02632, 0},{0,0.6830,0,0.02096},{1.0005,0,-0.000499,0}, {0,  0.8004,0,0.1996}};
  //arma::vec Bq0 = {{0.1320},{0.1402},{0},{0}};
  //arma::mat Gq0 = { {0.0774, 0, 0,0}, {0 , 0.0774,0,0}, {0,0,0.3872,0}, {0,0,0,0.3098}};
  //arma::vec Qq0 = {{3.3378},{2.9272}, {13.0207}, {10.4166}};
  //ssmodels_t model(Aq0,Bq0,Qq0,Gq0);

  arma::mat Aq0 = { {0.6682, 0},{0,0.6830}};
  arma::vec Bq0 = {{0.1320},{0.1402}};
  arma::mat Gq0 = { {0.0774, 0}, {0 , 0.0774}};
  arma::vec Qq0 = {{3.4364},{2.9272}};
  ssmodels_t model(Aq0,Bq0,Qq0,Gq0);


  std::vector<ssmodels_t> models1 = {model};
  shs_t<arma::mat,int> cs1SHS(models1);

  // Define max error
  double eps =.5;

  // Define safe set
  arma::mat Safe = {{19.5, 20.5}, {19.5,20.5}};

  arma::mat Input = {{18, 19}};

  // Define grid type
  // (1 = uniform, 2 = adaptive)
  // For comparison with IMDP we use uniform grid
  int gridType =2;

  // Time horizon
  int K = 6;

  // Library (1 = simulator, 2 = faust^2, 3 = imdp)
  int lb = 2;

  // Property type
  // FIX (1 = verify safety, 2= verify reach-avoid, 3 = safety synthesis, 4 = reach-avoid synthesis)
  int p = 1;

  // Task specification
  taskSpec_t cs1SpecFAUST(lb, K, p, Safe, Input,  eps,gridType,1);

  // Combine model and associated task
  inputSpec_t<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

  // Perform  Task
  performTask(cs1InputFAUST);


  // Perform  Task
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "------------Completed -----------"
            << std::endl;
}
