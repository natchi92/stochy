/*
 * ExtDatat.h
 *
 *  Created on: 11 Jan 2018
 *      Author: nathalie
 */

#include <armadillo>

#ifndef EXTDATAT_H_
#define EXTDATAT_H_
// To setup store for reading external
// inputs and initial
// values for both continous variables
// and discrete locations for the
// simulator

class exdata_t {
public:
  std::vector<arma::mat> X;      // Continuous variables: dim - x_dim,T
  std::optional<arma::mat> U;    // Exogenous inputs: dim - u_dim,T
  std::optional<arma::mat> D;    // Disturbance signals: dim - d_dim,T
  std::vector<arma::mat> q_init; // initial  value discrete modes
  std::optional<arma::mat> x_k;  // continuous variable evolution
  std::optional<arma::mat> u_k;  // input variable evolution
  std::optional<arma::mat> d_k;  // disturbance variable evolution
  std::vector<arma::mat> InitTq; // initial disturbance

public:
  exdata_t()
      : U{std::nullopt}, D{std::nullopt}, x_k{std::nullopt}, u_k{std::nullopt},
        d_k{std::nullopt} {}

  exdata_t(std::vector<arma::mat> X_in, arma::mat U_in, arma::mat D_in,
           std::vector<arma::mat> q_init_in)
      : X{X_in}, U{U_in}, D{D_in}, q_init{q_init_in} {
    x_k = X[0];
    u_k = U.row(0);
    d_k = D.row(0);
    InitTq = {arma::eye<arma::mat>(size(X[0]))};
  }
  exdata_t(std::vector<arma::mat> X_in, arma::mat U_in, arma::mat D_in,
           std::vector<arma::mat> q_init_in, std::vector<arma::mat> InitTq_in)
      : X{X_in}, U{U_in}, D{D_in}, q_init{q_init_in}, InitTq{InitTq_in} {
    x_k = X[0];
    u_k = U.row(0);
    d_k = D.row(0);
  }

  exdata_t(std::vector<arma::mat> X_in, arma::mat U_in,
           std::vector<arma::mat> q_init_in)
      : X{X_in}, U{U_in}, D{std::nullopt}, d_k{std::nullopt}, q_init{
                                                                  q_init_in} {

    x_k = X[0];
    u_k = U.row(0);
    InitTq = q_init_in;
  }

  exdata_t(std::vector<arma::mat> X_in, arma::mat Ud_in,
           std::vector<arma::mat> q_init_in, std::vector<arma::mat> InitTq_in,
           bool Noinput)
      : X{X_in}, q_init{q_init_in}, InitTq{InitTq_in} {
    x_k = X[0];
    if (Noinput) {
      U = std::nullopt;
      u_k = std::nullopt;
      D = Ud_in;
      d_k = Ud_in.row(0);
    } else {
      U = Ud_in;
      u_k = Ud_in.row(0);
      D = std::nullopt;
      d_k = std::nullopt;
    }
  }

  exdata_t(std::vector<arma::mat> X_in, std::vector<arma::mat> q_init_in,
           std::vector<arma::mat> InitTq_in)
      : X{X_in}, U{std::nullopt}, u_k{std::nullopt}, D{std::nullopt},
        d_k{std::nullopt}, q_init{q_init_in}, InitTq{InitTq_in}

  {
    x_k = X[0];
  }

  exdata_t(std::vector<arma::mat> X_in, std::vector<arma::mat> q_init_in)
      : X{X_in}, U{std::nullopt}, u_k{std::nullopt}, D{std::nullopt},
        d_k{std::nullopt}, q_init{q_init_in} {
    x_k = X[0];
    InitTq = {arma::eye<arma::mat>(X[0].n_rows, X[0].n_cols)};
  }

  virtual ~exdata_t(){};
};

#endif /* EXTDATAT_H_ */
