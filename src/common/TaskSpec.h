/*
 * TaskSpec.h
 *
 *  Created on: 17 Feb 2018
 *      Author: nathalie
 */

#ifndef TASKSPEC_H_
#define TASKSPEC_H_
#include <armadillo>
#include "StochEnums.h"

struct FAUSTInputs
{
  // synthesis FAUST related inputs
  Mat safeSet;
  Mat targetSet;
  Mat inputSet;
  double eps;
  int typeGrid;          // 1- Uniform Grid, 2- Local Adaptive, 3 - MCMC approx
  int assumptionsKernel; // 1- Lipschitz via Integral, 2- Lipschitz via sample,
                         // 3- Max-min
  int Controlled;
};

struct IMDPInputs
{
  Mat boundary;
  Mat Gridsize;
  Mat reftol;
};

struct SimulatorInputs
{
  int horizon;    // Time horizon to perform tool
  int runs; // Number of runs
};

template<typename T>
class TaskSpec
{
  TaskSpec(const PropertySpec _spec, const Library _tool, const T _inputParams)
          : spec(_spec), tool(_tool), inputParams(_inputParams) {}

public:
  Property spec;
  Library tool;
  T inputParams;

  virtual ~TaskSpec;
};

#endif // TASKSPEC_H_
