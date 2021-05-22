/*
 * SSModels.h
 *
 *  Created on: 10 Jan 2018
 *      Author: nathalie
 */

#ifndef SSMODELS_H_
#define SSMODELS_H_

#include "ExtDatat.h"
//#include "matio.h" /*Reading .arma::mat files*/
#include "utility.h"
#include <armadillo>
#include <functional>
#include <optional>

typedef arma::mat (*ssmodels_func)(unsigned n, const double *x, void *fdata);

typedef ssmodels_func func; // nlopt::func synoynm

class ssmodels_t {

public:
  ssmodels_t()
      : A{std::nullopt}, B{std::nullopt}, C{std::nullopt}, F{std::nullopt},
        N{std::nullopt}, G{std::nullopt}, Sigma{std::nullopt}, delta_t{1} {};

  ssmodels_t(std::optional<arma::mat> _A, std::optional<arma::mat> _B,
             std::optional<arma::mat> _C, std::optional<arma::mat> _F,
             std::optional<arma::mat> _N, std::optional<arma::mat> _G,
             std::optional<arma::mat> _Sigma, double _delta_t);

  // common model definition interface
  ssmodels_t(arma::mat _A, arma::mat _G);
  ssmodels_t(arma::mat _A, arma::mat _B, arma::mat _Sigma);
  ssmodels_t(arma::mat _A, arma::mat _B, arma::mat _N, arma::mat _Sigma);
  ssmodels_t(arma::mat _A, arma::mat _B, arma::mat _Q, arma::mat _F,
             arma::mat _Sigma);

  // initialise models from matfiles TODO(ncauchi) keep this feature? (matlab is
  // being dropped)
  void obtainSSfromMat(const char *fn, std::optional<int> currentMode);
  void populateStateSpaceModelFromMatFile(matvar_t &content, int currentMode);
  arma::mat fillMatrix(matvar_t &content);
  std::string readCells(matvar_t &content);

  // simple useful getter and setter functions
  std::optional<arma::mat> getAMatrix() { return A; }

  void setSampling(double _delta_t) { delta_t = _delta_t; }
  void setSigma(arma::mat _sigma) { Sigma = std::make_optional(_sigma); }

  // checking constructed model
  void checkModelStructure();
  void printModelStructure();

  // propogate model with time for simulator
  arma::mat getNextStateFromCurrent(arma::mat &x_k,
                                    std::optional<arma::mat> u_k,
                                    std::optional<arma::mat> g_k);

  ~ssmodels_t(){};

private:
  double delta_t; // If <= 0 then in CT model

  std::optional<arma::mat> A;
  std::optional<arma::mat> B;
  std::optional<arma::mat> C;
  std::optional<arma::mat>
      F; // F for disturbances of in case of switching systems to correspond to
         // the G function in Gw[k]
  std::optional<arma::mat> N; // N for bilinear models
  std::optional<arma::mat> G;
  std::optional<arma::mat> Sigma; // If Zeros then deterministic models
};

#endif /* SSMODELS_H_ */
