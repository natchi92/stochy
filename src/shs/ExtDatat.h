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

// TODO(ncauchi) convert into a struct and make it optionally defined
class exdata_t {
public:
  std::vector<arma::mat> X;      // Continuous variables: dim - x_dim,T
  arma::mat U;                   // Exogenous inputs: dim - u_dim,T
  arma::mat D;                   // Disturbance signals: dim - d_dim,T
  std::vector<arma::mat> q_init; // initial  value discrete modes
  arma::mat x_k;                 // continuous variable evolution
  arma::mat u_k;                 // input variable evolution
  arma::mat d_k;                 // disturbance variable evolution
  std::vector<arma::mat> InitTq; // initial disturbance

public:
  exdata_t() {
    X = {arma::zeros<arma::mat>(1, 10000)};
    U = arma::zeros<arma::mat>(1, 10000);
    D = arma::zeros<arma::mat>(1, 10000);
    q_init = {arma::zeros<arma::mat>(1, 10000)};
    x_k = X[0];
    u_k = U;
    d_k = D;
    InitTq = {arma::zeros<arma::mat>(1, 10000)};
  }
  exdata_t(std::vector<arma::mat> X_in, arma::mat U_in, arma::mat D_in,
           arma::mat x_max_in, arma::mat x_min_in,
           std::vector<arma::mat> q_init_in) {
    X = X_in;
    U = U_in;
    D = D_in;
    q_init = q_init_in;
    x_k = X_in[0];
    u_k = U_in.row(0);
    d_k = D_in.row(0);
    //  dW_k=dW_in.row(1);
    InitTq = {arma::eye<arma::mat>(size(X_in[0]))};
  }
  exdata_t(std::vector<arma::mat> X_in, arma::mat U_in, arma::mat D_in,
           std::vector<arma::mat> q_init_in, std::vector<arma::mat> InitTq_in) {
    X = X_in;
    U = U_in;
    D = D_in;
    q_init = q_init_in;
    x_k = X_in[0];
    u_k = U_in.row(0);
    d_k = D_in.row(0); //  dW_k=dW_in.row(1);
    InitTq = InitTq_in;
  }
  exdata_t(std::vector<arma::mat> X_in, arma::mat Ud_in,
           std::vector<arma::mat> q_init_in) {
    X = X_in;
    q_init = q_init_in;
    U = Ud_in;
    u_k = Ud_in.row(0);
    D = -1 * arma::ones<arma::mat>(1, 1);
    d_k = -1 * arma::ones<arma::mat>(1, 1);
    x_k = X_in[0];
    //  dW_kd_in.row(1);
    InitTq = q_init_in;
    d_k.reset();
  }

  exdata_t(std::vector<arma::mat> X_in, arma::mat Ud_in,
           std::vector<arma::mat> q_init_in, std::vector<arma::mat> InitTq_in,
           bool Noinput) {
    X = X_in;
    if (Noinput) {
      U = -1 * arma::ones<arma::mat>(1, 1);
      u_k = -1 * arma::ones<arma::mat>(1, 1);
      u_k.reset();
      D = Ud_in;
      d_k = Ud_in.row(0);
    } else {
      U = Ud_in;
      u_k = Ud_in.row(0);
      D = -1 * arma::ones<arma::mat>(1, 1);
      d_k = -1 * arma::ones<arma::mat>(1, 1);
      d_k.reset();
    }
    q_init = q_init_in;
    x_k = X_in[0];
    //  dW_kd_in.row(1);
    InitTq = InitTq_in;
  }
  exdata_t(std::vector<arma::mat> X_in, std::vector<arma::mat> q_init_in,
           std::vector<arma::mat> InitTq_in)

  {
    X = X_in;
    U = -1 * arma::ones<arma::mat>(1, 1);
    q_init = q_init_in;
    x_k = X_in[0];
    u_k = -1 * arma::ones<arma::mat>(1, 1);
    d_k = -1 * arma::ones<arma::mat>(1, 1);
    InitTq = InitTq_in;
    d_k.reset();
    u_k.reset();
  }

  exdata_t(std::vector<arma::mat> X_in, std::vector<arma::mat> q_init_in) {
    X = X_in;
    U = -1 * arma::ones<arma::mat>(1, 1);
    D = -1 * arma::ones<arma::mat>(1, 1);
    q_init = q_init_in;
    x_k = X_in[0];
    u_k = -1 * arma::ones<arma::mat>(1, 1);
    d_k = -1 * arma::ones<arma::mat>(1, 1);
    double nr = X_in[0].n_rows;
    double nc = X_in[0].n_cols;
    InitTq = {arma::eye<arma::mat>(nr, nc)};
    d_k.reset();
    u_k.reset();
  }

  virtual ~exdata_t(){};
};

#endif /* EXTDATAT_H_ */
