/*
 * TaskSpec.cpp
 *
 *  Created on: 17 Feb 2018
 *      Author: nathalie  TaskSpec(int inputTask);
 *
 */

#include "TaskSpec.h"

taskSpec_t::taskSpec_t() {
  // Initialise task to be performed
  // By default set  to simulation
  task = 1;
  T = 1; // default time horizon: 1 step
  runs = 10000; // default number of monte carlo simulations
  propertySpec = 1; // default property spec: simulation
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = 0;
  typeGrid = 1;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

taskSpec_t::taskSpec_t(int inputTask, int N) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = 1;
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = 0;
  typeGrid = 1;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
taskSpec_t::taskSpec_t(int inputTask, int N, int monte) {
  task = inputTask;
  T = N;
  runs = monte;
  propertySpec = 1;
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = 0;
  typeGrid = 1;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       arma::mat target, arma::mat input, double error,
                       int gridType) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
  safeSet = safety;
  targetSet = target;
  inputSet = input;
  eps = error;
  typeGrid = gridType;
  assumptionsKernel = 1;
  Controlled = 1;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat set,
                       arma::mat input, double error, int gridType,
                       int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
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
  inputSet = input;
  eps = error;
  typeGrid = gridType;
  assumptionsKernel = 1;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       double error, int gridType) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
  safeSet = safety;
  if (property != 1) {
    std::cout << "Incorrect definition of sets or specification" << std::endl;
    exit(0);
  }
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = error;
  typeGrid = gridType;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       double error, int gridType, int Kernel) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
  safeSet = safety;
  if (property != 1) {
    std::cout << "Incorrect definition of sets or specification" << std::endl;
    exit(0);
  }
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = error;
  typeGrid = gridType;
  assumptionsKernel = Kernel;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat set,
                       arma::mat input, double error, int gridType, int Kernel,
                       int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
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
  eps = error;
  typeGrid = gridType;
  assumptionsKernel = Kernel;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       arma::mat target, arma::mat input, double error,
                       int gridType, int Kernel, int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
  if ((property == 1) & (control == 1)) {
    safeSet = safety;
    inputSet = input;
    targetSet = arma::zeros<arma::mat>(1, 1);
  } else if ((property == 2) & (control == 0)) {
    safeSet = safety;
    targetSet = target;
    inputSet = arma::zeros<arma::mat>(1, 1);
  } else if ((property == 2) & (control == 1)) {
    safeSet = safety;
    targetSet = target;
    inputSet = input;

  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
  inputSet = input;
  eps = error;
  typeGrid = gridType;
  assumptionsKernel = Kernel;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat safety,
                       arma::mat target, arma::mat input, double error,
                       int gridType, int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
  if ((property == 1) & (control == 1)) {
    safeSet = safety;
    inputSet = input;
    targetSet = arma::zeros<arma::mat>(1, 1);
  } else if ((property == 2) & (control == 0)) {
    safeSet = safety;
    targetSet = target;
    inputSet = arma::zeros<arma::mat>(1, 1);
  } else if ((property == 2) & (control == 1)) {
    safeSet = safety;
    targetSet = target;
    inputSet = input;

  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
  eps = error;
  typeGrid = gridType;
  assumptionsKernel = 1;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

taskSpec_t::taskSpec_t(int inputTask, int N, int property, arma::mat bound,
                       arma::mat grid, arma::mat ref) {
  task = inputTask;
  T = N;
  runs = 10000;
  propertySpec = property;
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = -1;
  typeGrid = 0;
  assumptionsKernel = 0;
  Controlled = 0;
  boundary = bound;
  gridsize = grid;
  reftol = ref;
}

taskSpec_t::~taskSpec_t() {}
