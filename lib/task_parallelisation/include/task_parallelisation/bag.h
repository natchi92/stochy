#pragma once
#include <vector>

template <typename T>
class bag {
public:
    virtual ~bag() = 0;

    virtual T    next()  = 0;    
    virtual bool empty() = 0;
};


template <typename T>
class vector_bag : public bag<T> {
    std::vector<T>* tasks;

public:
    vector_bag(const std::vector<T>& tasks);
    vector_bag(const T* tasks, int n);
    ~vector_bag();

    T    next();
    bool empty();
};

template <typename T>
bag<T>::~bag() { }

template <typename T>
vector_bag<T>::vector_bag(const std::vector<T>& tasks) { 
    this -> tasks = new std::vector<T>(tasks);
}

template <typename T>
vector_bag<T>::vector_bag(const T* tasks, int n) { 
    this -> tasks = new std::vector<T>(tasks, tasks + n);
}

template <typename T>
vector_bag<T>::~vector_bag() { 
    if(tasks != NULL)
        delete tasks;
}

template <typename T>
T vector_bag<T>::next() {
    T &element = tasks -> back();
    tasks -> pop_back();
    return element;
} 

template <typename T>
bool vector_bag<T>::empty() {
    return tasks -> empty();
} 
