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
	if (argc >= 4)
		iterationNum = atoi(argv[3]);
	if (argc >= 5)
		eps = atof(argv[4]);
	
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

	readFile(argv[5], bmdp);

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

	unsigned int stateNum = bmdp.getNumStates(); 
	std::vector<MDP::State> vecState;

	// add states to IMC with cost 0 every where and cost 1 at the terminal states
	for (unsigned int i = 0; i < stateNum; i++)
	{
		vecState.push_back(MDP::State(i));
		imc.addState(&vecState[i],bmdp.getCost(i));
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
