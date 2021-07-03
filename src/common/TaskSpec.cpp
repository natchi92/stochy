/*
 * TaskSpec.cpp
 *
 *  Created on: 17 Feb 2018
 *      Author: nathalie  TaskSpec(int inputTask);
 *
 */

#include "TaskSpec.h"

taskSpec_t::taskSpec_t()
    : task{1}, T{1}, runs{5000}, propertySpec{1}, safeSet{std::nullopt},
      targetSet{std::nullopt}, inputSet{std::nullopt}, eps{1}, typeGrid{1},
      assumptionsKernel{1}, Controlled{0}, boundary{std::nullopt},
      gridsize{std::nullopt}, reftol{std::nullopt} {}

taskSpec_t::taskSpec_t(int inputTask, int N)
    : task{inputTask}, T{N}, runs{5000}, propertySpec{1}, safeSet{std::nullopt},
      targetSet{std::nullopt}, inputSet{std::nullopt}, eps{1}, typeGrid{1},
      assumptionsKernel{1}, Controlled{0}, boundary{std::nullopt},
      gridsize{std::nullopt}, reftol{std::nullopt} {}

taskSpec_t::taskSpec_t(int inputTask, int N, int monte)
    : task{inputTask}, T{N}, runs{monte}, propertySpec{1},
      safeSet{std::nullopt}, targetSet{std::nullopt}, inputSet{std::nullopt},
      eps{1}, typeGrid{1}, assumptionsKernel{1}, Controlled{0},
      boundary{std::nullopt}, gridsize{std::nullopt}, reftol{std::nullopt} {}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       arma::mat target, arma::mat input, double error,
                       int gridType)
    : task{inputTask}, T{N}, runs{5000}, propertySpec{property},
      safeSet{safety}, targetSet{target}, inputSet{target}, eps{error},
      typeGrid{gridType}, assumptionsKernel{1}, Controlled{0},
      boundary{std::nullopt}, gridsize{std::nullopt}, reftol{std::nullopt} {}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat set,
                       arma::mat input, double error, int gridType, int control)
    : task{inputTask}, T{N}, runs{5000}, propertySpec{property},
      safeSet{std::nullopt}, targetSet{std::nullopt}, inputSet{std::nullopt},
      eps{error}, typeGrid{gridType}, assumptionsKernel{1}, Controlled{control},
      boundary{std::nullopt}, gridsize{std::nullopt}, reftol{std::nullopt} {

  if ((property == 1) & (control == 1)) {
    safeSet = set;
    inputSet = input;
  } else if ((property == 2) & (control == 0)) {
    safeSet = set;
    targetSet = input;
  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       double error, int gridType)
    : task{inputTask}, T{N}, runs{5000}, propertySpec{property},
      safeSet{safety}, targetSet{std::nullopt}, inputSet{std::nullopt},
      eps{error}, typeGrid{gridType}, assumptionsKernel{1}, Controlled{0},
      boundary{std::nullopt}, gridsize{std::nullopt}, reftol{std::nullopt} {

  if (property != 1) {
    std::cout << "Incorrect definition of sets or specification" << std::endl;
    exit(0);
  }
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       double error, int gridType, int kernel)
    : task{inputTask}, T{N}, runs{5000}, propertySpec{property},
      safeSet{safety}, targetSet{std::nullopt}, inputSet{std::nullopt},
      eps{error}, typeGrid{gridType}, assumptionsKernel{kernel}, Controlled{0},
      boundary{std::nullopt}, gridsize{std::nullopt}, reftol{std::nullopt} {

  if (property != 1) {
    std::cout << "Incorrect definition of sets or specification" << std::endl;
    exit(0);
  }
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       arma::mat input, double error, int gridType, int kernel,
                       int control)
    : task{inputTask}, T{N}, runs{5000}, propertySpec{property},
      safeSet{safety}, targetSet{std::nullopt}, inputSet{input}, eps{error},
      typeGrid{gridType}, assumptionsKernel{kernel}, Controlled{control},
      boundary{std::nullopt}, gridsize{std::nullopt}, reftol{std::nullopt} {

  if ((property == 1) & (control == 1)) {
    safeSet = set;
    inputSet = input;
    targetSet = arma::zeros<arma::mat>(1, 1);

  } else if ((property == 2) & (control == 0)) {
    safeSet = set;
    targetSet = input;
    inputSet = arma::zeros<arma::mat>(1, 1);
  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat bound,
                       arma::mat grid, arma::mat ref)
    : task{inputTask}, T{N}, runs{5000},
      propertySpec{property}, safeSet{std::nullopt}, targetSet{std::nullopt},
      inputSet{std::nullopt}, eps{-1}, typeGrid{0}, assumptionsKernel{0},
      Controlled{0}, boundary{bound}, gridsize{grid}, reftol{ref} {}

taskSpec_t::~taskSpec_t() {}
