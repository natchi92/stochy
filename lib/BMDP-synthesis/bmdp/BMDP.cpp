#include "bmdp/BMDP.hpp"

using namespace MDP;

#include <iomanip>

const std::set<Action::ActionID>  BMDP::NO_ACTIONS     = std::set<Action::ActionID>();
const std::vector<State::StateID> BMDP::NO_STATES      = std::vector<State::StateID>();
const std::pair<double, double>   BMDP::NO_PROBABILITY = std::make_pair<double, double>(-1, -1);

MDP::BMDP::BMDP()
{
}

MDP::BMDP::~BMDP()
{
    clear();
}

void MDP::BMDP::addState(State* st, double cost)
{
    states_.insert(std::pair<State::StateID, State*>(st->getID(), st));
    cost_[st->getID()] = cost;
}

void MDP::BMDP::addAction(Action* act)
{
    actions_.insert(std::pair<Action::ActionID, Action*>(act->getID(), act));
}

MDP::State* MDP::BMDP::getState(const State::StateID& id) const
{
    std::map<State::StateID, State*>::const_iterator it = states_.find(id);
    if (it != states_.end())
        return it->second;

    return NULL;
}

MDP::Action* MDP::BMDP::getAction(const Action::ActionID& id) const
{
    std::map<Action::ActionID, Action*>::const_iterator it = actions_.find(id);
    if (it != actions_.end())
        return it->second;

    return NULL;
}

void MDP::BMDP::addTransition(State::StateID st, Action::ActionID act, State::StateID dest,
                              double plow, double phigh)
{
    // Make sure the states exist
    if (states_.find(st) == states_.end())
    {
        std::cout << "[Warning] Initial state with id " << st << " does NOT exist.  NOT adding transition." << std::endl;
        return;
    }
    if (states_.find(dest) == states_.end())
    {
        std::cout << "[Warning] Destination state with id" << dest << " does NOT exist.  NOT adding transition." << std::endl;
        return;
    }

    // Make sure the action exists
    if (actions_.find(act) == actions_.end())
    {
        std::cout << "[Warning] Action with id " << act << " does NOT exist.  NOT adding transition." << std::endl;
        return;
    }

    // Check if the transition already exists
    if (transitions_.find(Transition(st, act, dest)) != transitions_.end())
    {
        std::cout << "[Warning] Transition from state " << st << " using action " << act
                  << " to destination " << dest << " already exists.  NOT adding transition." << std::endl;
        return;
    }

    if (plow > phigh)
        throw std::runtime_error("Lower bound on probability is > than upper bound");

    // Adding the transition
    transitions_[Transition(st, act, dest)] = ProbInterval(plow, phigh);

    // Updating resulting states map
    std::vector<State::StateID>& result = resultingStates_[Result(st, act)];
    result.push_back(dest);

    // Updating state action list
    std::set<Action::ActionID>& actions = actionList_[st];
    actions.insert(act);
}

void MDP::BMDP::updateTransition(State::StateID st, Action::ActionID act, State::StateID dest,
                                 double plow, double phigh)
{
    if(!transitionExists(st, act, dest))
    {
        std::cout << "[Warning] Transition from state " << st << " using action " << act
                  << " to destination " << dest << " does not exist.  Cannot update something that never was." << std::endl;
        return;
    }

    // Updating the transition
    transitions_[Transition(st, act, dest)] = ProbInterval(plow, phigh);
}

void MDP::BMDP::clear()
{
    transitions_.clear();
    resultingStates_.clear();
    actionList_.clear();

    /*
    for(std::map<State::StateID, State*>::iterator it = states_.begin(); it != states_.end(); ++it)
        delete it->second;
    */
    states_.clear();


    /*
    for(std::map<Action::ActionID, Action*>::iterator it = actions_.begin(); it != actions_.end(); ++it)
        delete it->second;
    */
    actions_.clear();
    cost_.clear();
}

const std::set<Action::ActionID>& MDP::BMDP::getActionsAtState(State::StateID st) const
{
    boost::unordered_map<State::StateID, std::set<Action::ActionID> >::const_iterator it;
    it = actionList_.find(st);

    if (it == actionList_.end())
        return NO_ACTIONS;

    return it->second;
}

const std::vector<State::StateID>& MDP::BMDP::getDestinations(State::StateID st, Action::ActionID act) const
{
    boost::unordered_map<Result, std::vector<State::StateID>, ResultHash, ResultHash>::const_iterator it;
    it = resultingStates_.find(Result(st, act));

    if (it == resultingStates_.end())
        return NO_STATES;

    return it->second;
}

const std::pair<double, double>& MDP::BMDP::getProbabilityInterval(State::StateID st, Action::ActionID act, State::StateID dest) const
{
    boost::unordered_map<Transition, ProbInterval, TransitionHash, TransitionHash>::const_iterator it;
    it = transitions_.find(Transition(st, act, dest));

    if (it == transitions_.end())
        return NO_PROBABILITY;

    return it->second;
}

bool MDP::BMDP::transitionExists(State::StateID st, Action::ActionID act, State::StateID dest) const
{
    boost::unordered_map<Transition, ProbInterval, TransitionHash, TransitionHash>::const_iterator it;
    it = transitions_.find(Transition(st, act, dest));

    return it != transitions_.end();
}

unsigned int MDP::BMDP::getNumStates(void) const
{
    return states_.size();
}

unsigned int MDP::BMDP::getNumActions(void) const
{
    return actions_.size();
}

unsigned int MDP::BMDP::getNumActions(State::StateID st) const
{
    return getActionsAtState(st).size();
}

unsigned int MDP::BMDP::getNumDestinations(State::StateID st) const
{
    const std::set<Action::ActionID>& actions = getActionsAtState(st);
    unsigned int numDestinations = 0;

    std::set<Action::ActionID>::iterator it;
    for(it = actions.begin(); it != actions.end(); ++it)
        numDestinations += getNumDestinations(st, *it);

    return numDestinations;
}

unsigned int MDP::BMDP::getNumDestinations(State::StateID st, Action::ActionID act) const
{
    return getDestinations(st, act).size();
}

double MDP::BMDP::getCost(const State::StateID& st) const
{
    std::map<State::StateID, double>::const_iterator it = cost_.find(st);
    if (it == cost_.end())
        return std::numeric_limits<double>::infinity();

    return it->second;
}

MDP::State::StateID MDP::BMDP::getMostLikelyDestinationOptimistic(State::StateID st, Action::ActionID act) const
{
    const std::vector<State::StateID>& dest = getDestinations(st, act);
    State::StateID mlState = std::numeric_limits<State::StateID>::max();
    double maxProb = -1.0;

    for (size_t i = 0; i < dest.size(); ++i)
    {
        const std::pair<double, double>& prob = getProbabilityInterval(st, act, dest[i]);
        if (prob.second > maxProb)
        {
            maxProb = prob.second;
            mlState = dest[i];
        }
    }

    return mlState;
}


MDP::State::StateID MDP::BMDP::getMostLikelyDestinationPessimistic(State::StateID st, Action::ActionID act) const
{
    const std::vector<State::StateID>& dest = getDestinations(st, act);
    State::StateID mlState = std::numeric_limits<State::StateID>::max();
    double maxProb = -1.0;

    for (size_t i = 0; i < dest.size(); ++i)
    {
        const std::pair<double, double>& prob = getProbabilityInterval(st, act, dest[i]);
        if (prob.first > maxProb)
        {
            maxProb = prob.first;
            mlState = dest[i];
        }
    }

    return mlState;
}

bool MDP::BMDP::isValid(void) const
{
    bool valid = true;

    for (size_t i = 0; i < states_.size() && valid; ++i)
    {
        const std::set<Action::ActionID>& actions = getActionsAtState(i);
        std::set<Action::ActionID>::const_iterator it;
        for (it = actions.begin(); it != actions.end() && valid; ++it)
        {
            const std::vector<State::StateID>& destinations = getDestinations(i, *it);

            double lbprob = 0.0;
            double ubprob = 0.0;

            for (size_t k = 0; k < destinations.size() && valid; ++k)
            {
                const std::pair<double, double>& probInterval = getProbabilityInterval(i, *it, destinations[k]);
                lbprob += probInterval.first;
                ubprob += probInterval.second;

                // lower bound probability is > upper bound
                if (probInterval.first > probInterval.second)
                {
                    valid = false;
                    std::cout << "Invalid BMDP (state " << i << " action " << *it << "): lower bound > upper bound  ["
                              << probInterval.first << " " << probInterval.second << "]" << std::endl;
                }
            }

            // Sum of lower bound probabilities must be <= 1.0
            if (lbprob > 1.0 && fabs(1.0 - lbprob) > 1e-6)
            {
                valid = false;
            //    std::cout << "Invalid BMDP (state " << i << " action " << *it << "): Sum of lower bound must be <= 1.0 [" << lbprob << "]" << std::endl;
            }

            // Sum of upper bound probabilities must be >= 1.0
            if (ubprob < 1.0 && fabs(1.0 - ubprob) > 1e-6)
            {
                valid = false;
            //    std::cout << "Invalid BMDP (state " << i << " action " << *it << "): Sum of upper bound must be >= 1.0 [" << ubprob << "]" << std::endl;
            }
        }
    }

    return valid;
}

void MDP::BMDP::makeProbabilitiesValid(void)
{
    for (size_t i = 0; i < states_.size(); ++i)
    {
        const std::set<Action::ActionID>& actions = getActionsAtState(i);
        std::set<Action::ActionID>::const_iterator it;
        for (it = actions.begin(); it != actions.end(); ++it)
        {
            const std::vector<State::StateID>& destinations = getDestinations(i, *it);

            double lbprob = 0.0;
            double ubprob = 0.0;

            for (size_t k = 0; k < destinations.size(); ++k)
            {
                const std::pair<double, double>& probInterval = getProbabilityInterval(i, *it, destinations[k]);
                lbprob += probInterval.first;
                ubprob += probInterval.second;
            }

            if (lbprob > 1.0 && fabs(1.0 - lbprob) > 1e-6)
            {
                // Normalizing all probabilities by lbprob
                for (size_t k = 0; k < destinations.size(); ++k)
                {
                    const std::pair<double, double>& probInterval = getProbabilityInterval(i, *it, destinations[k]);
                    transitions_[Transition(i, *it, destinations[k])] = ProbInterval(probInterval.first/lbprob, probInterval.second/lbprob);
                }
            }
            if (ubprob < 1.0 && fabs(1.0 - ubprob) > 1e-6)
            {
                // Normalizing all probabilities by lbprob
                for (size_t k = 0; k < destinations.size(); ++k)
                {
                    const std::pair<double, double>& probInterval = getProbabilityInterval(i, *it, destinations[k]);
                    transitions_[Transition(i, *it, destinations[k])] = ProbInterval(probInterval.first/ubprob, probInterval.second/ubprob);
                }
            }
        }
    }

    std::cout << "After making probabilites valid, BMDP is " << (isValid() ? "VALID" : " STILL INVALID!") << std::endl;
}

void MDP::BMDP::drawPolicy(const MDP::BMDP::Policy& policy, std::ostream& o) const
{
    o << std::setprecision(6) << "digraph MarkovChain{" << std::endl;

    // Defining all states
    std::map<State::StateID, State*>::const_iterator stateit;
    for (stateit = states_.begin(); stateit != states_.end(); ++stateit)
    {
        std::map<MDP::State::StateID, double>::const_iterator it = cost_.find(stateit->second->getID());

        if (it == cost_.end())
            throw std::runtime_error("Failed to find cost value for state");

        double cost = it->second;
        if (cost < 0)
            o << stateit->second->getID() << " [color=blue] [shape=box]" << std::endl;
        else
            o << stateit->second->getID() << std::endl;
    }

    // Adding transitions
    for (stateit = states_.begin(); stateit != states_.end(); ++stateit)
    {
        MDP::BMDP::Policy::const_iterator it = policy.find(stateit->second->getID());

        if (it == policy.end())
            throw std::runtime_error("Failed to find action for a state in the policy");

        MDP::Action::ActionID act = it->second;
        const std::vector<State::StateID>& destinations = getDestinations(stateit->second->getID(), act);
        for (size_t j = 0; j < destinations.size(); ++j)
        {
            const std::pair<double, double>& probability = getProbabilityInterval(stateit->second->getID(), act, destinations[j]);
            o << std::setprecision(6) << stateit->second->getID() << " -> " << destinations[j] << " [label=\"(" << probability.first << " " << probability.second << ")\"]" << std::endl;
        }
    }

    o << "}" << std::endl;
}
