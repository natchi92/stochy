#pragma once
#include "bag.h"
#include <vector>
#include <thread>
#include <mutex>

#define DEFAULT_THREAD_POOL_SIZE 40

template <typename T, typename R>
class task_manager_base {
protected:
    bag<T>* m_bag;

    std::function<R(T)> task_solver;
    
    int thread_pool_size;
    std::vector<std::thread> thread_pool;
    std::mutex get_task_mutex;

    void start_workers();
    virtual void worker() = 0;
    bool get_task(T &task);

public:
    task_manager_base(bag<T>* m_bag, 
                 std::function<R(T)> task_solver,
                 int thread_pool_size = DEFAULT_THREAD_POOL_SIZE);
    task_manager_base(const std::vector<T>& tasks, 
                 std::function<R(T)> task_solver,
                 int thread_pool_size = DEFAULT_THREAD_POOL_SIZE);
    task_manager_base(const T* tasks, int n, 
                 std::function<R(T)> task_solver,
                 int thread_pool_size = DEFAULT_THREAD_POOL_SIZE);
    ~task_manager_base();
};

template <typename T, typename R>
class task_manager : public task_manager_base<T, R> {
    std::vector<std::pair<T, R>> result;
    std::mutex add_result_mutex;

    void worker();

public:
    using task_manager_base<T, R>::task_manager_base;
    std::vector<std::pair<T, R>> run();
};

template <typename T>
class task_manager<T, void> : public task_manager_base<T, void> {
    void worker();
   
public:
    using task_manager_base<T, void>::task_manager_base;
    void run();
};


// Implementing generic base class

template <typename T, typename R> 
task_manager_base<T, R>::task_manager_base( bag<T>* m_bag, 
                                            std::function<R(T)> task_solver,
                                            int thread_pool_size)
                                : m_bag(m_bag), task_solver(task_solver),
                                  thread_pool_size(thread_pool_size) { }

template <typename T, typename R> 
task_manager_base<T, R>::task_manager_base( const std::vector<T>& tasks, 
                                            std::function<R(T)> task_solver,
                                            int thread_pool_size)
                                : task_solver(task_solver),
                                  thread_pool_size(thread_pool_size) { 
    m_bag = new vector_bag<T>(tasks);
}

template <typename T, typename R> 
task_manager_base<T, R>::task_manager_base( const T* tasks, int n, 
                                            std::function<R(T)> task_solver,
                                            int thread_pool_size)
                                : task_solver(task_solver),
                                  thread_pool_size(thread_pool_size) { 
    m_bag = new vector_bag<T>(tasks, n);
}

template <typename T, typename R> 
task_manager_base<T, R>::~task_manager_base(){
    thread_pool.clear();
    if(m_bag != NULL) {      
        delete m_bag;
        m_bag = NULL;
    }
}

template <typename T, typename R> 
void task_manager_base<T, R>::start_workers(){
    for(int i = 0; i < thread_pool_size; i++)
        thread_pool.push_back(std::thread([&](){
            worker();
        }));
}

template <typename T, typename R> 
bool task_manager_base<T, R>::get_task(T& task){
    get_task_mutex.lock();
    if(m_bag -> empty()) {
        get_task_mutex.unlock();
        return false;
    }
    task = m_bag -> next();
    get_task_mutex.unlock();
    return true;
}


// Implementing generic child class 

template <typename T, typename R> 
std::vector<std::pair<T, R>> task_manager<T, R>::run(){
    this -> start_workers();
    this -> worker();
    for(int i = 0; i < this -> thread_pool_size; i++)
        this -> thread_pool[i].join();
    return result;
}

template <typename T, typename R> 
void task_manager<T, R>::worker(){
    T task{};
    while(true){
        if(!this -> get_task(task))
            return;
        R res = this -> task_solver(task);

        add_result_mutex.lock();
        result.push_back({task, res});
        add_result_mutex.unlock();
    }
}


// Implementing partial-specialisation of child class

template <typename T> 
void task_manager<T, void>::worker(){
    T task{};
    while(true){
        if(!this -> get_task(task))
            return;
        this -> task_solver(task);
    }
}

template <typename T> 
void task_manager<T, void>::run(){
    this -> start_workers();
    this -> worker();
    for(int i = 0; i < this -> thread_pool_size; i++)
        this -> thread_pool[i].join();
}

