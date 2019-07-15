#include <bmdp/BMDP.hpp>
#include <bmdp/IntervalValueIteration.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

void readFile(const char* filename, MDP::BMDP &bmdp)
{
	std::vector<MDP::State> vecState;
	std::vector<MDP::Action> vecAction;
	unsigned stateNum, actNum, termState, termNum, cur, act, dest;
	std::vector<unsigned int> stateTerm;
	double pmin, pmax;
	std::ifstream fin (filename);

	if (!fin)
	{
		std::cout << "Could NOT open the file" << std::endl;
		return;
	}
	
	// read the first 3 numbers
	fin >> stateNum >> actNum >> termNum;

	//std::cout << "StateNum = " << stateNum << "; actNum = " << actNum << "; termNum = " << termNum << std::endl;

	// read terminal states
	for (unsigned int i =0; i< termNum; i++)
	{
		fin >> termState;
		//std::cout << termState << std::endl;
		stateTerm.push_back(termState);
	}

	// add bmdp states with the correct reward
	for (unsigned int i=0; i<stateNum; i++)
	{
		vecState.push_back(MDP::State(i));

		bool termFlag = false;
		for (unsigned int j=0 ; j<stateTerm.size(); j++)
		{
			if (stateTerm[j] == i)
			{
				termFlag = true;
				break;
			}
		}

		if (termFlag)
			bmdp.addState(&vecState[i],1.0);
		else
			bmdp.addState(&vecState[i],0.0);
	}

	// add actions
	for (unsigned int i =0; i < actNum; i++)
	{
		vecAction.push_back(MDP::Action(i));
		bmdp.addAction(&vecAction[i]);
	}

	// add transitions
	while (!fin.eof())
	{
		fin >> cur;
		if (fin.eof())
			break;

		fin >> act >> dest >> pmin >> pmax;

		bool termFlag = false;

		//fin >> cur >> act >> dest >> pmin >> pmax;

		for (unsigned int i = 0; i < stateTerm.size(); i++)
		{
			if (stateTerm[i] == cur)
			{
				termFlag = true;
				break;
			}
		}

		if (!termFlag)
			bmdp.addTransition(cur, act, dest, pmin, pmax);
	}
	fin.close();
}

int main(int argc, char **argv)
{
	double eps = 1e-4;
	int iterationNum = -1;
	bool maxPolicy = true; //true = maximization and false = minimization
	
	if (argc < 2)
	{
		std::cout << std::endl;
		std::cout << "**** NEED BMDP MODEL ****" << std::endl;
		std::cout << "inputs: (BMDP model file name) NumberOfIterations(optional) epsilon(optional)" << std::endl;
		std::cout << std::endl;
		return 0;
	}

	/*if (strcmp(argv[1] , "maximize") && strcmp(argv[1] , "minimize"))
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
		maxPolicy = false;*/
	if (argc >= 3)
		iterationNum = atoi(argv[2]);
	if (argc >= 4)
		eps = atof(argv[3]);
	
/*	std::cout << std::endl;
	std::cout << "******************************" << std::endl;
	if (maxPolicy)
		std::cout << "maxPolicy = " << "maximize" << std::endl;
	else
		std::cout << "maxPolicy = " << "minimize" << std::endl;

	std::cout << "policyType = " << policyType << std::endl;
	std::cout << "Number of Iterations = " << iterationNum << std::endl;
	std::cout << "eps = " << eps << std::endl;
	std::cout << "******************************" << std::endl;
	std::cout << std::endl;*/

	
	//___________________________________________________________________________________________________
	// SETTING UP BMDP MODEL
	//---------------------------------------------------------------------------------------------------
	MDP::BMDP bmdp;

	readFile(argv[1], bmdp);

	// check if bmpd is valid
	bmdp.isValid();
	//---------------------------------------------------------------------------------------------------

	//___________________________________________________________________________________________________
	// BMDP INTERVAL VALUE ITERATION
	//---------------------------------------------------------------------------------------------------
	MDP::IntervalValueIteration ivi(bmdp);

	MDP::BMDP::Policy policy;
	std::vector<double> minVals;
	std::vector<double> maxVals;

	// set the dicount rate
	ivi.setDiscount(1.);

	// compute maxVals for the policy that maximizes the upper bound
	ivi.computeOptimisticPolicy(policy, maxPolicy, maxVals, maxVals, iterationNum, eps);

	// compute minVals for the policy that minimizes the lower bound
	ivi.computeOptimisticPolicy(policy, !maxPolicy, minVals, minVals, iterationNum, eps);
	//--------------------------------------------------------------------------------------------------
	

	// print policy & bounds
	std::cout << std::endl;
	for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end(); it++)
	{
		int mu = it->second;

		std::cout << it->first << " " << mu << " " << minVals[it->first] << " " << maxVals[it->first] << std::endl;
		//std::cout << it->first << " " << mu << " [" << minVals[it->first] << "," << maxVals[it->first] << "]" << std::endl;		
		//std::cout << bmdp.getCost(it->first) << std::endl;
	}
	std::cout << std::endl;
	return 0;
}