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

  // Initial continuous variable
  // number of continuous variables
  int num_c = 2;
  arma::mat x_init = arma::ones<arma::mat>(num_c, monte);

  // Initialise random generators
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());

  // normal distribution defined with a mean and a variance
  // d{mean, variance}
  std::normal_distribution<double> d1{450, 25};
  std::normal_distribution<double> d2{17, 2};
  for (size_t i = 0; i < monte; ++i) {
    x_init(0, i) = d1(generator);
    x_init(1, i) = d2(generator);
  }

  std::vector<arma::mat> X = {x_init};

  // Initial discrete mode q_0 = (E,C)
  int q0 = 0; // 0 = (E,C), 1 = (F,C), 2 = (F,O), 3 = (E,O)
  std::vector<arma::mat> q_init = {q0*arma::ones<arma::mat>(T1 + 1, monte)};

  // Definition of control signal
  // Read from .txt/.mat file or define here as defined x_init
  arma::mat U = readInputSignal("u.txt");
  exdata_t data(X, U, q_init);

  // Get model from file
  shs_t<arma::mat, int> cs4SHS("CS4.mat", data);

  // To specify model manually
  // comment the previous line of code
  // i.e. shs_t<arma::mat,int> cs4SHS("CS4.mat'")
  // and then uncomment the following lines of code
  // which allow you to manually define the model
  // dynamics for each mode
  // ---------------------------------------------
  /*  // Definition of Transition matrix
   arma::mat Tq= {{0.6, 0.4, 0, 0},{0.1, 0.15, 0.75, 0},{0, 0.25, 0.65, 0.1},{0, 0, 0.4, 0.6}};

  // For mode q_0 = (E,C)
   arma::mat Aq0 = {{0.9957, 0},{2.6354e-5, 0.9868}};
   arma::vec Bq0 = {{0},{0.9268}};
   arma::mat Nq0 = {{-0.0152, 0},{0, -0.0463}};
   arma::vec Fq0 = {{1.6211},{0.0012}};
   arma::mat Gq0 = {{58.095, 0},{0, 3.97}};
   ssmodels_t modelq0(Aq0,Bq0,Nq0,Fq0,Gq0);

  // For mode q_1 = (F,C)
   arma::mat Aq1 = {{0.9957, 0},{2.6354e-5, 0.9868}};
   arma::vec Bq1 = {{0},{0.9268}};
   arma::mat Nq1 = {{-0.0152, 0},{0, -0.0463}};
   arma::vec Fq1 = {{7.6211},{0.3177}};
   arma::mat Gq1 = {{58.095, 0},{0, 3.97}};
   ssmodels_t modelq1(Aq1,Bq1,Nq1,Fq1,Gq1);
  //
  // For mode q_2 = (F,O)
   arma::mat Aq2 = {{0.9995, 0},{2.6354e-5, 0.9984}};
   arma::vec Bq2 = {{0},{0.9268}};
   arma::mat Nq2 = {{-0.0152, 0},{0, -0.0463}};
   arma::vec Fq2 = {{6.1953},{0.0393}};
   arma::mat Gq2 = {{58.095, 0},{0, 3.97}};
   ssmodels_t modelq2(Aq2,Bq2,Nq2,Fq2,Gq2);
  //
  // For mode q_3 = (E,O)
   arma::mat Aq3 = {{0.9995, 0},{2.6354e-5, 0.9984}};
   arma::vec Bq3 = {{0},{0.9268}};
   arma::mat Nq3 = {{-0.0152, 0},{0, -0.0463}};
   arma::vec Fq3 = {{0.1953},{0.0012}};
   arma::mat Gq3 = {{58.095, 0},{0, 3.97}};
   ssmodels_t modelq3(Aq3,Bq3,Nq3,Fq3,Gq3);

   std::vector<ssmodels_t> models = {modelq0, modelq1, modelq2, modelq3};
   shs_t<arma::mat,int> cs4SHS(Tq, models, data);
  // --------------------------------------------*/

  // Library (1 - simulator, 2 - faust^2, 3 = imdp)
  int lb = 1;
  // Specify task to be performed
  taskSpec_t cs4Spec(lb, T1, monte);

  // Combine model and associated task
  inputSpec_t<arma::mat, int> cs4Input(cs4SHS, cs4Spec);

  // Perform Task
  performTask(cs4Input);

  std::cout << "----------------------------------------" << std::endl;
  std::cout << "------------Completed Case Study 4 : Results in results "
               "folder -----------"
            << std::endl;
}
