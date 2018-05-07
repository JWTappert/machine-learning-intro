x = 12
learn = 0.1
precision = 0.00001
prev_step = 1/precision

# f = lambda x: (x ** 2) - (2 * x) + 1
df = lambda x: (2 * x) - 2

while prev_step > precision:
    prev_x = x
    x += -learn * df(prev_x)
    prev_step = abs(x - prev_x)
    
print("min: %s" % x)
