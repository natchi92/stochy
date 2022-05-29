/*
 * InputSpec.h
 *
 *  Created on: 17 Feb 2018
 *      Author: nathalie
 */

#ifndef INPUTSPEC_H_
#define INPUTSPEC_H_

#include "SHS.h"
#include "TaskSpec.h"

template <typename T, typename T2>
class InputSpec
{
  InputSpec(): stochastic(true), sigmoid(false), multiple_init_states(false) {}
  InputSpec(bool _stochastic,
            bool _sigmoid,
            bool _multi,
            SHS<T, T2> model,
            TaskSpec task)
           :  stochastic(_stochastic), sigmoid(_sigmoid), multiple_init_states(_multi),
              myModel(model), myTask(task)
           {}

  InputSpec(SHS<T, T2> model, TaskSpec task)
           : stochastic(true), sigmoid(false), multiple_init_states(false),
              myModel(model), myTask(task)
           {}
public:
  bool stochastic; // identify whether system is deterministic or not
  bool sigmoid; // To indicate to use the sigmoid function as the discrete transition kernel
  bool multiple_init_states;  // To indicate multiple initial states

  SHS<T, T2> myModel; // obtain state space model
  TaskSpec myTask; // store task to be performed

  virtual ~InputSpec() {}
};
#endif
