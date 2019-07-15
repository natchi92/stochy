#ifndef MDP_ACTION_HPP
#define MDP_ACTION_HPP

/// Definition of an action in a Markov Decision Process
/// Requires a unique ID
namespace MDP
{
    class Action
    {
        public:
            typedef unsigned int ActionID;

            Action(ActionID id) : id_(id) {}
            virtual ~Action() {}

            virtual ActionID getID() const { return id_; }

        protected:
            ActionID id_;
    };
}

#endif
