import numpy as np

def sdense(s, n=300): return(int(np.ceil(460 * n * (1-s) + n + n**2 * (1-s) + n + n**2 * (1-s) + n + n)))

def dense(n): return(460 * n + n + n**2 + n + n**2 + n + n)


print 'Dense network', sdense(0.)
print 'sparsity 0.8', sdense(0.8)

print 'dense', dense(98)

print 'sparsity 0.90', sdense(0.9)
print 'dense', dense(57)
print 'sparsity 0.95', sdense(0.95)
print 'dense', dense(32)

print 'sparsity 0.98', sdense(0.98)
print 'dense', dense(16)


print 'sparsity 0.99', sdense(0.99)
print 'dense', dense(9)


print 'sparsity 0.995', sdense(0.995)
print 'dense', dense(6)

