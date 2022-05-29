/*
 * TaskSpec.cpp
 *
 *  Created on: 17 Feb 2018
 *      Author: nathalie  TaskSpec(int inputTask);
 *
 */

#include "TaskSpec.h"

TaskSpec::TaskSpec()
{
  // Initialise task to be performed
  // By default set  to simulation
  task = 1;
  T = 1; // default time horizon: 1 step
  runs = 10000; // default number of monte carlo simulations
  PropertySpec = 1; // default Property spec: simulation
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = 0;
  typeGrid = 1;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

TaskSpec::TaskSpec(int inputTask, int N) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = 1;
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = 0;
  typeGrid = 1;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
TaskSpec::TaskSpec(int inputTask, int N, int monte) {
  task = inputTask;
  T = N;
  runs = monte;
  PropertySpec = 1;
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = 0;
  typeGrid = 1;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat safety,
                       arma::mat target, arma::mat input, double error,
                       int GridType) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  safeSet = safety;
  targetSet = target;
  inputSet = input;
  eps = error;
  typeGrid = GridType;
  assumptionsKernel = 1;
  Controlled = 1;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat set,
                       arma::mat input, double error, int GridType,
                       int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  if ((Property == 1) & (control == 1)) {
    safeSet = set;
    inputSet = input;
    targetSet = arma::zeros<arma::mat>(1, 1);
  } else if ((Property == 2) & (control == 0)) {
    safeSet = set;
    targetSet = input;
    inputSet = arma::zeros<arma::mat>(1, 1);
  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
  inputSet = input;
  eps = error;
  typeGrid = GridType;
  assumptionsKernel = 1;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat safety,
                       double error, int GridType) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  safeSet = safety;
  if (Property != 1) {
    std::cout << "Incorrect definition of sets or specification" << std::endl;
    exit(0);
  }
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = error;
  typeGrid = GridType;
  assumptionsKernel = 1;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat safety,
                       double error, int GridType, int Kernel) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  safeSet = safety;
  if (Property != 1) {
    std::cout << "Incorrect definition of sets or specification" << std::endl;
    exit(0);
  }
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = error;
  typeGrid = GridType;
  assumptionsKernel = Kernel;
  Controlled = 0;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat set,
                       arma::mat input, double error, int GridType, int Kernel,
                       int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  if ((Property == 1) & (control == 1)) {
    safeSet = set;
    inputSet = input;
    targetSet = arma::zeros<arma::mat>(1, 1);

  } else if ((Property == 2) & (control == 0)) {
    safeSet = set;
    targetSet = input;
    inputSet = arma::zeros<arma::mat>(1, 1);
  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
  eps = error;
  typeGrid = GridType;
  assumptionsKernel = Kernel;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}
TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat safety,
                       arma::mat target, arma::mat input, double error,
                       int GridType, int Kernel, int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  if ((Property == 1) & (control == 1)) {
    safeSet = safety;
    inputSet = input;
    targetSet = arma::zeros<arma::mat>(1, 1);
  } else if ((Property == 2) & (control == 0)) {
    safeSet = safety;
    targetSet = target;
    inputSet = arma::zeros<arma::mat>(1, 1);
  } else if ((Property == 2) & (control == 1)) {
    safeSet = safety;
    targetSet = target;
    inputSet = input;

  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
  inputSet = input;
  eps = error;
  typeGrid = GridType;
  assumptionsKernel = Kernel;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat safety,
                       arma::mat target, arma::mat input, double error,
                       int GridType, int control) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  if ((Property == 1) & (control == 1)) {
    safeSet = safety;
    inputSet = input;
    targetSet = arma::zeros<arma::mat>(1, 1);
  } else if ((Property == 2) & (control == 0)) {
    safeSet = safety;
    targetSet = target;
    inputSet = arma::zeros<arma::mat>(1, 1);
  } else if ((Property == 2) & (control == 1)) {
    safeSet = safety;
    targetSet = target;
    inputSet = input;

  } else {
    std::cout << "Incorrect definition of sets" << std::endl;
    exit(0);
  }
  eps = error;
  typeGrid = GridType;
  assumptionsKernel = 1;
  Controlled = control;
  boundary = arma::zeros<arma::mat>(1, 1);
  Gridsize = arma::zeros<arma::mat>(1, 1);
  reftol = arma::zeros<arma::mat>(1, 1);
}

TaskSpec::TaskSpec(int inputTask, int N, int Property, arma::mat bound,
                       arma::mat Grid, arma::mat ref) {
  task = inputTask;
  T = N;
  runs = 10000;
  PropertySpec = Property;
  safeSet = arma::zeros<arma::mat>(1, 1);
  targetSet = arma::zeros<arma::mat>(1, 1);
  inputSet = arma::zeros<arma::mat>(1, 1);
  eps = -1;
  typeGrid = 0;
  assumptionsKernel = 0;
  Controlled = 0;
  boundary = bound;
  Gridsize = Grid;
  reftol = ref;
}

TaskSpec::~TaskSpec() {}
