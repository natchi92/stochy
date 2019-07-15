#include <stdio.h>
#include <stdlib.h>
//#include <dlfcn.h>
#include "bmdp/BMDP.hpp"
#include "bmdp/IntervalValueIteration.hpp"
 
int main(int argc, char **argv){

  char* policyType;
  double eps = 1e-4;
  int iterationNum = -1;
  bool maxPolicy = true;

  if (argc < 3)
  {
    std::cout << std::endl;
    std::cout << "**** need to specify a policy type (maximiz/minimize pessimistic/optimistic) ****" << std::endl;
    std::cout << "inputs: (maximize/minimize) (pessimistic/optimistic) NumberOfIterations(optional) epsilon(optional)" << std::endl;
    std::cout << std::endl;
    return 0;
  }

  if (strcmp(argv[1] , "maximize") && strcmp(argv[1] , "minimize"))
  {
    std::cout << std::endl;
    std::cout << "**** maximize policy OR minimize policy? ****" << std::endl;
    std::cout << std::endl;
    return 0;
  }

  if (strcmp(argv[2] , "pessimistic") && strcmp(argv[2] , "optimistic"))
  {
    std::cout << std::endl;
    std::cout << "**** policy types are pessimistic or optimistic ****" << std::endl;
    std::cout << std::endl;
    return 0;
  }

  policyType = argv[2];
  if (argc >= 3 && (strcmp(argv[1], "minimize") ==0) )
    maxPolicy = false;
  if(argc >= 4)
  {
    iterationNum = atoi(argv[3]);
  } 
  if(argc >= 5)
  {
    eps = atof(argv[4]);
  }

  //  void *handle;
  //  char *error;
   // int x, y, z;
 
    /*handle = dlopen ("libsynth.so.1", RTLD_LAZY);
    if (!handle) {
        fputs (dlerror(), stderr);
        exit(1);
    }
 
    dlsym(handle, "MDP::BMDP::BMDP()");
    if (( error = dlerror() ) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }
 
    printf("managed to load dll");
      */ 
//___________________________________________________________________________________________________
  // SETTING UP BMDP MODEL
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP bmdp;

  unsigned int stateNum = 3;
  unsigned int actionNum = 2;
  std::vector<MDP::State> vecState;
  std::vector<MDP::Action> vecAction;
  MDP::BMDP::Policy policy;

  // add states to bmdp with cost 0 every where and cost 1 at the terminal states
  for (unsigned int i = 0; i < stateNum; i++)
  {
    vecState.push_back(MDP::State(i));

    if(i<stateNum-1)
      bmdp.addState(&vecState[i]);
    else
      bmdp.addState(&vecState[i],1.0);
  }

  // add actions to bmpd
  for (unsigned int i =0; i < actionNum; i++)
  {
    vecAction.push_back(MDP::Action(i));
    bmdp.addAction(&vecAction[i]);
  }

  // add the tranistion probabilities to bmdp
  for (unsigned int i =0; i <stateNum -1 ;i++)
  {
    for(unsigned int j=0; j<actionNum; j++)
    {
      for(unsigned int k=0; k<stateNum;k++)
      {
        //bmdp.addTransition(i, j, k, .1, .9);
        if (i==0 && k==1 && j == 0)
          bmdp.addTransition(i, j, k, .1, .3);
        else if (i==0 && k==2 && j == 0)
          bmdp.addTransition(i, j, k, .8, .9);
        else if (i==1 && k==0 && j == 0)
          bmdp.addTransition(i, j, k, .5, .7);
        else if (i==1 && k==2 && j == 0)
          bmdp.addTransition(i, j, k, .4, .6);
        else if (i==2 && k==0 && j == 0)
          bmdp.addTransition(i, j, k, .8, 1.);
        else if (i==2 && k==1 && j == 0)
          bmdp.addTransition(i, j, k, .1, .3);
        else if (i==0 && k==1 && j == 1)
          bmdp.addTransition(i, j, k, .7, .8);
        else if (i==0 && k==2 && j == 1)
          bmdp.addTransition(i, j, k, .2, .95);
      }
    }
  } 

 // check if bmpd is valid
  bmdp.isValid();
  //---------------------------------------------------------------------------------------------------

  //___________________________________________________________________________________________________
  // BMDP INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi(bmdp);

  std::vector<double> minVals;
  std::vector<double> maxVals;

  // set the dicount rate
  ivi.setDiscount(1.);

  //compute policy.  maxVals and minVals are the upper and lower bounds of transitin probabilties
  if (strcmp(policyType,"pessimistic") == 0)
    ivi.computePessimisticPolicy(policy, maxPolicy, minVals, minVals, iterationNum, eps);
  else
    ivi.computeOptimisticPolicy(policy, maxPolicy, maxVals, maxVals, iterationNum, eps);
  //--------------------------------------------------------------------------------------------------


  //___________________________________________________________________________________________________
  // SETTING UP IMC
  // build an IMC (new BMDP) with the computed policy to find the lower bounds for optimistic or upper bounds for pessimistic
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP imc;

  // add states to IMC with cost 0 every where and cost 1 at the terminal states
  for (unsigned int i = 0; i < stateNum; i++)
  {
    vecState.push_back(MDP::State(i));

    if(i<stateNum-1)
      imc.addState(&vecState[i]);
    else
      imc.addState(&vecState[i],1.0);
  }

  // add actions
  MDP::Action imcAction = 0;
  imc.addAction(&imcAction);

  // add the tranistion probabilities to imc
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end(); it++)
  {
    std::vector<MDP::State::StateID> dest = bmdp.getDestinations(it->first, it->second);
    for (unsigned int i = 0; i < dest.size(); i++)
    {
      std::pair<double, double> probs = bmdp.getProbabilityInterval(it->first, it->second, dest[i]);
      imc.addTransition(it->first, 0, dest[i], probs.first, probs.second);
    }
  }

  //___________________________________________________________________________________________________
  // IMC INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi_imc(imc);

  MDP::BMDP::Policy imcPolicy;

  // set the dicount rate
  ivi_imc.setDiscount(1.);

  //compute policy.  maxVals and minVals are the upper and lower bounds of transitin probabilties
  if (strcmp(policyType,"pessimistic") == 0)
    ivi_imc.computeOptimisticPolicy(imcPolicy, maxPolicy, maxVals, maxVals, iterationNum, eps);
  else
    ivi_imc.computePessimisticPolicy(imcPolicy, maxPolicy, minVals, minVals, iterationNum, eps);

   // print policy & bounds
  std::cout << std::endl;
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end(); it++)
  {
    int mu = it->second;

    std::cout << it->first << " " << mu << " " << minVals[it->first] << " " << maxVals[it->first] << std::endl;
   }
  std::cout << std::endl;
  //dlclose(handle);
  return 0;
}
