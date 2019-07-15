#ifndef MDP_BMDP_HPP
#define MDP_BMDP_HPP

#include "bmdp/MarkovProcess.hpp"

#include <set>
#include <map>
#include <vector>
#include <iostream>

#include <boost/unordered_map.hpp>

namespace MDP
{
    /// Definition of a bounded-parameter Markov decision process
    /// TODO: Change State and Action references to pointers to prevent object slicing
    class BMDP
    {
        public:
            typedef std::map<State::StateID, Action::ActionID> Policy;

            static const std::set<Action::ActionID>  NO_ACTIONS;
            static const std::vector<State::StateID> NO_STATES;
            static const std::pair<double, double>   NO_PROBABILITY;

            BMDP();
            ~BMDP();

            /// Add the given state to the BMDP with a specified cost (default zero)
            void addState(State* st, double cost = 0.0);
            /// Add the given action to the BMDP
            void addAction(Action* act);

            State* getState(const State::StateID& id) const;
            Action* getAction(const Action::ActionID& id) const;

            /// Add a transition from state st to state dest using action act with the
            /// given upper and lower bound probabilities
            /// If a transition for this triple exists, nothing will be changed
            void addTransition(State::StateID st, Action::ActionID act, State::StateID dest,
                               double plow, double phigh);

            /// Updates the existing transition for this triple.  If the transition does
            /// not exist, nothing will be changed
            void updateTransition(State::StateID st, Action::ActionID act, State::StateID dest,
                               double plow, double phigh);

            /// Clear all states and actions from the BMDP
            void clear();

            /// Retrieve the total number of states
            unsigned int getNumStates(void) const;
            /// Retrieve the total number of distinct actions
            unsigned int getNumActions(void) const;
            /// Retrieve the number of valid actions at state st
            unsigned int getNumActions(State::StateID st) const;
            /// Retrieve the total number of possible destinations at this state (over all
            /// mapped transitions)
            unsigned int getNumDestinations(State::StateID st) const;
            /// Retrieve the number of distinct destinations for which have a probability assigned for
            /// the given state-action pair.
            unsigned int getNumDestinations(State::StateID st, Action::ActionID act) const;
            /// Retrieve the cost associated with a particular state
            double getCost(const State::StateID& st) const;

            /// Return the set of valid actions for the given state
            const std::set<Action::ActionID>& getActionsAtState(State::StateID st) const;
            /// Return the set of possible destinations for the given state-action pair
            const std::vector<State::StateID>& getDestinations(State::StateID st, Action::ActionID act) const;
            /// Return the probability interval for the state-action-destination pair
            const std::pair<double, double>& getProbabilityInterval(State::StateID st, Action::ActionID act, State::StateID dest) const;
            /// Return true if a transition already exists between this state/action/destination triple
            bool transitionExists(State::StateID st, Action::ActionID act, State::StateID dest) const;

            /// Return the state with the maximum probability, wrt upper bound on transition probability
            State::StateID getMostLikelyDestinationOptimistic(State::StateID st, Action::ActionID act) const;
            /// Return the state with the maximum probability, wrt lower bound on transition probability
            State::StateID getMostLikelyDestinationPessimistic(State::StateID st, Action::ActionID act) const;

            /// Checks to see if this BMDP is a valid BMDP
            bool isValid(void) const;
            /// A hack function to normalize probabilities.  Ensures this BMDP is valid.
            void makeProbabilitiesValid(void);

            /// Draws a policy (Markov chain) using DOT format
            void drawPolicy(const Policy& policy, std::ostream& o=std::cout) const;

        protected:

            typedef std::pair<double, double> ProbInterval;


            /// A mapping of (s,a,s') to the probability of executing action a at state s and ending up at state s'.
            boost::unordered_map<Transition, ProbInterval, TransitionHash, TransitionHash>    transitions_;
            /// A mapping of (s,a) to the list of resulting states that the system could end up in
            boost::unordered_map<Result, std::vector<State::StateID>, ResultHash, ResultHash> resultingStates_;
            /// A mapping of states to the possible actions available
            boost::unordered_map<State::StateID, std::set<Action::ActionID> >                 actionList_;

            /// The list of states in the BMDP
            std::map<State::StateID, State*>    states_;
            /// The list of actions in the BMDP
            std::map<Action::ActionID, Action*> actions_;

            /// The cost function for states in the BMDP
            std::map<State::StateID, double>   cost_;
    };
}

#endif
