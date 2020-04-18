from pairhmm import *
import sys

Model( 0.5, 0.3, 0.5, 0.1, 4 )



pseudocounts = [1]*6

bw = BaumWelch( 0.5, 0.3, 0.5, 0.1, 4, *pseudocounts )


x = "ACGG"
y = "ACGG"

x = "ATCC"
y = "ATCC"


model = bw.build_model()

pair = SequencePair( model, x, y )

print("X:", x)
print("Y:", y)

log_likelihood = pair.forward_algorithm()

print("log likelihood:", log_likelihood)

pair.print_log_f()

pair.backward_algorithm()

pair.print_log_b()

data = [[x,y]]
param = model.estimate_parameters( data, *pseudocounts )
print(param)

bw.iterate(data)