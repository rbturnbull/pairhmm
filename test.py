from pairhmm import *

Model( 0.5, 0.3, 0.5, 0.1, 4 )


bw = BaumWelch( 0.5, 0.3, 0.5, 0.1, 4 )

x = "ACGTCC"
y = "AAGG"

model = bw.build_model()

pair = SequencePair( model, x, y )
print("X:", x)
print("Y:", y)

log_likelihood = pair.forward_algorithm()
print("log likelihood:", log_likelihood)

print("log f_A")
print(pair.log_f_A)

print("log f_I")
print(pair.log_f_I)

print("log f_D")
print(pair.log_f_D)