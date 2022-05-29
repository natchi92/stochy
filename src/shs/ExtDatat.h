/*
 * ExtDatat.h: To setup store for reading external inputs and initial
 * values for both continous variables and discrete locations for the
 * SIMULATOR
 *
 *  Created on: 11 Jan 2018
 *      Author: nathalie
 */

#ifndef EXTDATAT_H_
#define EXTDATAT_H_

#include <armadillo>

using VectorMatrices = std::vector<arma::mat>;
using OptionalMatrix = std::optional<arma::mat>;

struct ExData
{
  ExData(VectorMatrices& _X, OptionalMatrix& _U, OptionalMatrix& _D, VectorMatrices &_InitTq) :
         X(_X), U(_U), D(_D), InitTq(_InitTq), numSamples(10000), x_k(_X.front()) {}

  const void set_input_variable(OptionalMatrix& _u_k) { u_k = _u_k; }
  const void set_disturbance_variable(OptionalMatrix& _d_k) { d_k = _d_k; }

  VectorMatrices X;    // Continuous variables: dim - x_dim,T
  OptionalMatrix U;    // Exogenous inputs: dim - u_dim,T
  OptionalMatrix D;    // Disturbance signals: dim - d_dim,T
  VectorMatrices q_init; // initial  value discrete modes
  VectorMatrices InitTq; // initial disturbance

  OptionalMatrix x_k;  // continuous variable evolution
  OptionalMatrix u_k;  // input variable evolution
  OptionalMatrix d_k;  // disturbance variable evolution
  int numSamples;
};

#endif /* EXTDATAT_H_ */
