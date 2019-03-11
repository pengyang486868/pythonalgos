import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a=np.array([2,5,8,9],dtype=float)
b=a**2+1
dfa=pd.Series(a)
dfb=pd.Series(b)
#mat=dfa.append(dfb,ignore_index=True)
mat=pd.DataFrame({'a':dfa,'b':dfb})
plt.plot(a,b)
plt.show()
print('second')
