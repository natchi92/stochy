/*a
 * InputSpec.h
 *
 *  Created on: 17 Feb 2018
 *      Author: nathalie
 */

#ifndef INPUTSPEC_H_
#define INPUTSPEC_H_

#include "SHS.h"
#include "TaskSpec.h"

template <class T, class T2> class inputSpec_t {
public:
  bool stoch; // identify whether system is deterministic or not
  bool sigm; // To indicate to use the sigmoid function as the discrete
             // transition kernel
  bool par;  // To indicate multiple initial states
  shs_t<T, T2> myModel; // obtain state space model
  taskSpec_t myTask; // store task to be performed

public:
  inputSpec_t() {
    stoch = true; // assuming model is stocahstic i.e. not deterministic
    sigm = false;
    par = false;
    shs_t<T, T2> model;
    myModel = model;
    taskSpec_t task;
    myTask = task;
  }
  inputSpec_t(bool stochastic, bool sigmoid, bool multi, shs_t<T, T2> model,
              taskSpec_t task) {
    stoch = stochastic;
    sigm = sigmoid;
    par = multi;
    myModel = model;
    myTask = task;
  }
  inputSpec_t(shs_t<T, T2> model, taskSpec_t task) {
    stoch = true;
    sigm = false;
    par = false;
    myModel = model;
    myTask = task;
  }
  virtual ~inputSpec_t() {}
};
#endif
