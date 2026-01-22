# Block Scheduling

## First attempt
Instead of scheduling each sequence one by one at each leaf of the tree of the sequence creation:
```cpp
class BnBBlockOptimizer {
    inline auto add_elimination() {
        if (chain.is_accumulated) {
            #pragma omp task
            scheduler->schedule(sequence);
        }
    }
}:
```
We gather the sequences in an `std::vector` and then schedule them all at once:
```cpp
class BnBBlockOptimizer {
    inline auto add_elimination() {
        if (chain.is_accumulated) {
            #pragma omp critical
            sequences.push_back(sequence);
        }
    }

//...

    inline auto schedule_all() {
        #pragma omp target map(to:seqs[:n], scheduler)
        #pragma omp parallel for
        for (std::size_t i = 0; i < sequences.size(); i++) {
            scheduler->schedule(sequence)
        }
    }
};
```

## Second attempt
We gather the sequences as in the previous attempt, but here, we schedule them further into the scheduler implementation
```cpp
class BnBBlockOptimizer {
    inline auto schedule_all_late() {
        m_scheduler->schedule_gpu(sequences, ...);
    }
};
```
We need to gather all the data beforehand
```cpp
class BnBBlockScheduler {
    std::size_t schedule_gpu(sequences, ...) {
        // Prepare data
        std::vector<std::size_t> vec_usable_threads(sequences.size());
        std::vector<std::size_t> vec_sequential_makespan(sequences.size());
        std::vector<Sequence> vec_working_copy(sequences.size());
        std::vector<std::size_t> vec_best_makespan(sequences.size());
        std::vector<std::size_t> vec_thread_loads(threads);
        std::vector<std::size_t> vec_lower_bound(sequences.size());
        std::vector<std::size_t> results(sequences.size());
        std::size_t n = sequences.size();
        
        // Some work...
        
        // Schedule everything
        #pragma omp target map(to :seqs[:n], ut[:n], sms[:n], wc[:n], bms[:n], lbs[:n]) map(r[:n], tl[:n])
        #pragma omp parallel for
        for (std::size_t i = 0; i < n; i++) {
           r[i] = lambda_schedule(seqs[i], ut[i],
              wc[i], bms[i], tl,
              lbs[i], sms[i], upper_bound);
        }
    }
};
```

Using the lambda_schedule function:
```cpp
std::size_t lambda_schedule(sequence, ...) {
    // recursive lambda expression scheduler
    auto schedule_op = [&](auto& schedule_next_op) {}
}
```