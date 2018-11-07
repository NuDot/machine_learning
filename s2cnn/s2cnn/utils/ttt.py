from complex import as_complex
import numpy as np

x = np.arange(4)
x = x.reshape((2, 2))

print(x)
#real = np.ones_like[x]
y = as_complex(x)
print(y)

z = complex(x, y)
print(z)
