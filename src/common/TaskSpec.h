/*
 * TaskSpec.h
 *
 *  Created on: 17 Feb 2018
 *      Author: nathalie
 */

#ifndef TASKSPEC_H_
#define TASKSPEC_H_
#include <armadillo>

class taskSpec_t {
public:
  int task; // 1- Simulation, 2 - Verification FAUST, 3 - Verification  syntheis
            // BMDP
  int T;    // Time horizon to perform tool
  int runs; // Number of runs
  // Model checking related
  int propertySpec; // 1-safety, 2-reach and avoid, 3 - formula free (FAUST),
  // 1 - safety verification, 2 - safety synthesis, 3 - RA verification, 4 - RA
  // synthesis FAUST related inputs
  arma::mat safeSet;
  arma::mat targetSet;
  arma::mat inputSet;
  double eps;
  int typeGrid;          // 1- Uniform Grid, 2- Local Adaptive, 3 - MCMC approx
  int assumptionsKernel; // 1- Lipschitz via Integral, 2- Lipschitz via sample,
                         // 3- Max-min
  int Controlled;
  // BMDP related
  arma::mat boundary;
  arma::mat gridsize;
  arma::mat reftol;

public:
  taskSpec_t();
  taskSpec_t(int inputTask, int N);
  taskSpec_t(int inputTask, int N, int monte);
  taskSpec_t(int inputTask, int N, int property, arma::mat safety,
             arma::mat target, arma::mat input, double error, int gridType);
  taskSpec_t(int inputTask, int N, int property, arma::mat set, arma::mat input,
             double error, int gridType, int control);
  taskSpec_t(int inputTask, int N, int property, arma::mat safety, double error,
             int gridType);
  taskSpec_t(int inputTask, int N, int property, arma::mat safety, double error,
             int gridType, int Kernel);
  taskSpec_t(int inputTask, int N, int property, arma::mat set, arma::mat input,
             double error, int gridType, int Kernel, int control);
  taskSpec_t(int inputTask, int N, int property, arma::mat safety,
             arma::mat target, arma::mat input, double error, int gridType,
             int Kernel, int control);
  taskSpec_t(int inputTask, int N, int property, arma::mat safety,
             arma::mat target, arma::mat input, double error, int gridType,
             int control);
  taskSpec_t(int inputTask, int N, int property, arma::mat bound,
             arma::mat grid, arma::mat ref);
  virtual ~taskSpec_t();
};

#endif /* TASKSPEC_H_ */
