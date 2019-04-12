import timeit
# import sys
import numpy as np
# import pandas as pd

np.random.seed(99)

start = timeit.default_timer()



stop = timeit.default_timer()
total_time = stop - start

print ()
print ("FINISHED! Total time taken: " + str(total_time) + " seconds")