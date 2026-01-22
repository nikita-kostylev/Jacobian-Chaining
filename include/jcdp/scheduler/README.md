Here we will explain the general structure branch_and_bound_gpu.

A layer on the stack is scheduling an operation onto a thread.

If the lower bound is lower than the current best_markspan the current changes are stored on the stack and we go a level deeper.

After the changes ran we have three edge cases.

First is the case where we try the next thread on the same depth.

The second is where we try the next operation on the same depth starting with thread = 0.

The third one is where we reduce the depth by one to get a layer up.