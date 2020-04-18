from pairhmm import *

Model( 0.5, 0.3, 0.5, 0.1, 4 )


bw = BaumWelch( 0.5, 0.3, 0.5, 0.1, 4 )


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

print("log f_A")
print(pair.log_f_A)

print("log f_I")
print(pair.log_f_I)

print("log f_D")
print(pair.log_f_D)

pair.backward_algorithm()

print("log b_A")
print(pair.log_b_A)

print("log b_I")
print(pair.log_b_I)

print("log b_D")
print(pair.log_b_D)
