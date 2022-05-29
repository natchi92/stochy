/*
 * StochEnums.h - stores all the enums used in Stochy
 *
 *  Created on: 11 Jan 2018
 *      Author: nathalie
 */

#ifndef STOCHY_ENUMS_H_
#define STOCHY_ENUMS_H_

using Mat = arma::mat;

enum class ErrorOnDataStructure
{
  NO_ERROR,
  DATA_ERROR,
  X_DIM_ERROR,
  U_DIM_ERROR,
  D_DIM_ERROR
};

enum class Library
{
  SIMULATOR,
  MDP,
  IMDP
};

enum Grid
{
  UNIFORM,
  ADAPTIVE
};

enum Property
{
  VERIFY_SAFETY,
  VERIFY_REACH_AVOID,
  SAFETY_SYNTHESIS,
  REACH_AVOID_SYNTHESIS,
  FORMULA_FREE
};

#endif // STOCHY_ENUMS_H_
