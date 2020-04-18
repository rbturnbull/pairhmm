import numpy as np
np.get_include()

cimport numpy as np

#from sselogsumexp import logsumexp
from scipy.special import logsumexp

cdef (float) logsumexp_2( float a, float b ):
#    return logsumexp( np.asarray( [a,b], dtype=np.float32 ) )
#    return logsumexp( [a,b] )
    if a == np.NINF:
        return b
    if b == np.NINF:
        return a

    cdef float mymax = max(a, b)
    return mymax + np.log(np.exp(a - mymax) + np.exp(b - mymax))

cdef (float) logsumexp_3( float a, float b, float c ):
#    return logsumexp( np.asarray( [a,b,c], dtype=np.float32 ) )
    if a == np.NINF:
        return logsumexp_2(b, c)
    if b == np.NINF:
        return logsumexp_2(a, c)
    if c == np.NINF:
        return logsumexp_2(a, b)
        
#    return logsumexp( [a,b,c] )
    cdef float mymax = max(a, b, c)
    return mymax + np.log(np.exp(a - mymax) + np.exp(b - mymax) + np.exp(c - mymax))


cdef class Model():
    cpdef int alphabet_size
    
    cpdef float constant_log_q
    cpdef float log_p_match
    cpdef float log_p_mismatch
    
    cpdef float log_transition_A_A
    cpdef float log_transition_I_A
    cpdef float log_transition_D_A
    
    cpdef float log_transition_A_I
    cpdef float log_transition_A_D
    cpdef float log_transition_I_I
    cpdef float log_transition_D_D
    
    cpdef float log_transition_A_E
    cpdef float log_transition_I_E
    cpdef float log_transition_D_E
    
    
    cpdef float current_log_likelihood

    cpdef float current_expected_A_A
    cpdef float current_expected_A_I
    cpdef float current_expected_A_D
    
    cpdef float current_expected_I_A
    cpdef float current_expected_I_I

    cpdef float current_expected_D_A
    cpdef float current_expected_D_D
    
    cpdef float current_expected_matches
    cpdef float current_expected_mismatches
    
    
    
    def __init__(self, float p_match, float p_gap_start, float p_gap_extend, float p_end, int alphabet_size):
        self.alphabet_size = alphabet_size
        
        self.constant_log_q = np.log( 1.0/alphabet_size )        
        self.log_p_match = np.log( p_match )
        self.log_p_mismatch = np.log( (1.0 - p_match)/(alphabet_size - 1) )
        
        self.log_transition_A_A = np.log( 1.0 - 2.0*p_gap_start - p_end )
        self.log_transition_I_A = self.log_transition_D_A = np.log( 1.0 - p_gap_extend - p_end )
        
        self.log_transition_A_I = self.log_transition_A_D = np.log( p_gap_start )
        self.log_transition_I_I = self.log_transition_D_D = np.log( p_gap_extend )
        
        self.log_transition_A_E = self.log_transition_I_E = self.log_transition_D_E = np.log( p_end )

    cpdef (float) log_q( self, str item ):
        return self.constant_log_q
        
    cpdef (float) log_p( self, str itemX, str itemY ):
        return self.log_p_match if itemX == itemY else self.log_p_mismatch
    cpdef (float) log_likelihood(self):
        return self.current_log_likelihood

        
    def estimate_parameters( self, data, pseudocount_match, pseudocount_mismatch, pseudocount_align, pseudocount_gap_start, pseudocount_gap_end, pseudocount_gap_extend ):
        self.expected_values(data)
        
        matches = self.current_expected_matches + pseudocount_match
        mismatches = self.current_expected_mismatches + pseudocount_mismatch
        
        A_A = self.current_expected_A_A + pseudocount_align
        A_I = self.current_expected_A_I + pseudocount_gap_start
        A_D = self.current_expected_A_D + pseudocount_gap_start    
        
        D_A = self.current_expected_D_A + pseudocount_gap_end
        D_D = self.current_expected_D_D + pseudocount_gap_extend

        I_A = self.current_expected_I_A + pseudocount_gap_end
        I_I = self.current_expected_I_I + pseudocount_gap_extend        
        
        estimated_p_match = matches/(matches + mismatches)
        estimated_p_gap_start = 0.5 * (A_I + A_D)/(A_A + A_I + A_D)
        estimated_p_gap_extend = 0.5 * (I_I + D_D)/(I_I + D_D + I_A + D_A)
        
        return estimated_p_match, estimated_p_gap_start, estimated_p_gap_extend

    def expected_values( self, data ):   
        cdef int i, j
     
        self.current_log_likelihood = 0.0
    
        self.current_expected_A_A = 0.0
        self.current_expected_A_I = 0.0
        self.current_expected_A_D = 0.0
        
        self.current_expected_I_A = 0.0     
        self.current_expected_I_I = 0.0     

        self.current_expected_D_A = 0.0        
        self.current_expected_D_D = 0.0
        
        self.current_expected_matches = 0.0
        self.current_expected_mismatches = 0.0
        
        for index, (sequenceX,sequenceY) in enumerate(data):
            pair = SequencePair( self, sequenceX, sequenceY )
            log_probability_sequence = pair.forward_algorithm()
            pair.backward_algorithm()
                        
            if index % 10 == 0:
                print("Seq %d" % index)
            
            self.current_log_likelihood += log_probability_sequence
            
            for i in range( 1, pair.n ):
                for j in range( 1, pair.m ):
                    # Transition Expectations
                    self.current_expected_A_A += np.exp( pair.log_f_A[i,j] + self.log_transition_A_A + pair.log_p(i+1,j+1) + pair.log_b_A[i+1,j+1] - log_probability_sequence )
                    self.current_expected_A_I += np.exp( pair.log_f_A[i,j] + self.log_transition_A_I + pair.log_q_x( i+1 ) + pair.log_b_I[i+1,j]   - log_probability_sequence )
                    self.current_expected_A_D += np.exp( pair.log_f_A[i,j] + self.log_transition_A_D + pair.log_q_y( j+1 ) + pair.log_b_D[i,j+1]   - log_probability_sequence )
        
                    self.current_expected_I_A += np.exp( pair.log_f_I[i,j] + self.log_transition_I_A + pair.log_p(i+1,j+1) + pair.log_b_A[i+1,j+1] - log_probability_sequence )     
                    self.current_expected_I_I += np.exp( pair.log_f_I[i,j] + self.log_transition_I_I + pair.log_q_x( i+1 ) + pair.log_b_I[i+1,j]   - log_probability_sequence )     

                    self.current_expected_D_A += np.exp( pair.log_f_D[i,j] + self.log_transition_D_A + pair.log_p(i+1,j+1) + pair.log_b_A[i+1,j+1] - log_probability_sequence )        
                    self.current_expected_D_D += np.exp( pair.log_f_D[i,j] + self.log_transition_D_D + pair.log_q_y( j+1 ) + pair.log_b_D[i+1,j]   - log_probability_sequence )

                    # Emission Expectations
                    posterior_state_A = np.exp( pair.log_f_A[i,j] + pair.log_b_A[i,j] - log_probability_sequence )
                    if pair.sequence_item_X(i) == pair.sequence_item_Y(j):
                        self.current_expected_matches += posterior_state_A
                    else:
                        self.current_expected_mismatches += posterior_state_A
                    

cdef class SequencePair():
    cpdef Model model
    cpdef str sequenceX
    cpdef str sequenceY

    cpdef int n
    cpdef int m
    
    cpdef np.ndarray log_f_A
    cpdef np.ndarray log_f_I
    cpdef np.ndarray log_f_D

    cpdef float log_f_E
    
    cpdef np.ndarray log_b_A
    cpdef np.ndarray log_b_I
    cpdef np.ndarray log_b_D


    def __init__(self, Model model, str sequenceX, str sequenceY):
        self.model = model
        self.sequenceX = sequenceX
        self.sequenceY = sequenceY

        self.n = len(sequenceX)
        self.m = len(sequenceY)
        
        self.log_f_A = None
        self.log_f_I = None
        self.log_f_D = None

        self.log_f_E = np.NINF
        
        self.log_b_A = None
        self.log_b_I = None
        self.log_b_D = None
        
        
    cpdef (float) log_q_x( self, int i ):
        if i > self.n:
            return np.NINF
        return self.model.log_q( self.sequence_item_X(i) )
    cpdef (float) log_q_y( self, int j ):
        if j > self.m:
            return np.NINF
    
        return self.model.log_q( self.sequence_item_Y(j) )
        
    cpdef (str) sequence_item_X(self, int i):
        return self.sequenceX[ i-1 ]
    cpdef (str) sequence_item_Y(self, int j):
        return self.sequenceY[ j-1 ]
    
    cpdef (float) log_p( self, int i, int j ):
        if i > self.n or j > self.m:
            return np.NINF
    
        return self.model.log_p( self.sequence_item_X(i), self.sequence_item_Y(j) )

    def print_log_f(self):
        print("log f_A")
        print(self.log_f_A)

        print("log f_I")
        print(self.log_f_I)

        print("log f_D")
        print(self.log_f_D)
        
    def print_log_b(self):        
        print("log b_A")
        print(self.log_b_A)

        print("log b_I")
        print(self.log_b_I)

        print("log b_D")
        print(self.log_b_D)
        
    
    def forward_algorithm( self ):
        '''
        Calculates the combined probability of all alignments up to position (i,j) that end in a particular state.
        R. Durbin, S. Eddy, A. Krogh, G. Mitchison, Biological Sequence Analysis, 87.
        
        Returns the log probability of the whole sequence pair log(P(x,y)) i.e. log(f_E(n,m))
        '''
        
        # Check if this function has already been called for this sequence pair
        if self.log_f_A and self.log_f_I and self.log_f_D:
            return self.log_f_E
                      
        cdef int i, j
                    
        cpdef np.ndarray log_f_A = np.empty( (self.n+1, self.m+1), dtype=np.float32 )
        cpdef np.ndarray log_f_I = np.empty( (self.n+1, self.m+1), dtype=np.float32 )        
        cpdef np.ndarray log_f_D = np.empty( (self.n+1, self.m+1), dtype=np.float32 )
        
        ####################################
        # Initial values at start boundaries
        ####################################
        # f_A( 0,0 ) has a probability of 1 so the log is 0.0
        log_f_A[0,0] = 0.0
        log_f_I[0,0] = np.NINF
        log_f_D[0,0] = np.NINF

        # Initialize j == 0 boundary     
        if self.n > 0:  
            log_f_A[1,0] = np.NINF         
            log_f_I[1,0] = self.model.log_transition_A_I + self.log_q_x( 1 )
            log_f_D[1,0] = np.NINF
            
            for i in range( 2, self.n+1 ):
                log_f_A[i,0] = np.NINF            
                log_f_I[i,0] = self.log_q_x( i ) + self.model.log_transition_I_I + log_f_I[i-1,0]
                log_f_D[i,0] = np.NINF

        # Initialize i == 0 boundary  
        if self.m > 0:              
            log_f_A[0,1] = np.NINF               
            log_f_I[0,1] = np.NINF               
            log_f_D[0,1] = self.model.log_transition_A_D + self.log_q_y( 1 )
            for j in range( 2, self.m+1 ):
                log_f_A[0,j] = np.NINF
                log_f_I[0,j] = np.NINF
                log_f_D[0,j] = self.log_q_y( j ) + self.model.log_transition_D_D + log_f_D[0,j-1]
                
        ####################################
        # Recursion
        ####################################
        for i in range( 1, self.n+1 ):
            for j in range( 1, self.m+1 ):
                ### Probability ends in Alignment
                log_f_A[i,j] = self.log_p( i, j ) + logsumexp_3(
                    self.model.log_transition_A_A + log_f_A[i-1,j-1],
                    self.model.log_transition_I_A + log_f_I[i-1,j-1],
                    self.model.log_transition_D_A + log_f_D[i-1,j-1],
                    )
                    
                ### Probability ends in Insertion
                log_f_I[i,j] = self.log_q_x( i ) + logsumexp_2(
                    self.model.log_transition_A_I + log_f_A[i-1,j],
                    self.model.log_transition_I_I + log_f_I[i-1,j],
                    )                

                ### Probability ends in Deletion
                log_f_D[i,j] = self.log_q_y( j ) + logsumexp_2(
                    self.model.log_transition_A_D + log_f_A[i,j-1],
                    self.model.log_transition_D_D + log_f_D[i,j-1],
                    )
        
        ####################################
        # Termination
        ####################################
        self.log_f_E = logsumexp_3(
                    self.model.log_transition_A_E + log_f_A[self.n, self.m],
                    self.model.log_transition_I_E + log_f_I[self.n, self.m],
                    self.model.log_transition_D_E + log_f_D[self.n, self.m],
                    )
        
        self.log_f_A = log_f_A
        self.log_f_I = log_f_I
        self.log_f_D = log_f_D
        
        return self.log_f_E          
            
            
    def backward_algorithm( self ):
        '''
        Calculates the probability of the latter part of the alignment given the state at (i,j)
        R. Durbin, S. Eddy, A. Krogh, G. Mitchison, Biological Sequence Analysis, 93.        
        '''
        # Check if this function has already been called for this sequence pair        
        if self.log_b_A and self.log_b_I and self.log_b_D:
            return
        cdef int i, j        
        
        cpdef np.ndarray log_b_A = np.empty( (self.n+1, self.m+1), dtype=np.float32 ) # We don't calculate the 0 rows so this could be optimized but it is written like this for the sake of clarity
        cpdef np.ndarray log_b_I = np.empty( (self.n+1, self.m+1), dtype=np.float32 )        
        cpdef np.ndarray log_b_D = np.empty( (self.n+1, self.m+1), dtype=np.float32 )
        
        cdef float A_end
        cdef float I_end
        cdef float D_end
                
        ####################################
        # Initial values at start boundaries
        ####################################        
        log_b_A[self.n, self.m] = self.model.log_transition_A_E
        log_b_I[self.n, self.m] = self.model.log_transition_I_E
        log_b_D[self.n, self.m] = self.model.log_transition_D_E
        
        for i in range( self.n-1, 0, -1 ):
            j = self.m
            log_b_A[i,j] = self.model.log_transition_A_I + self.log_q_x(i+1) + log_b_I[i+1,j]            
            log_b_I[i,j] = self.model.log_transition_I_I + self.log_q_x(i+1) + log_b_I[i+1,j]
            log_b_D[i,j] = np.NINF
            
        for j in range( self.m-1, 0, -1 ):
            i = self.n       
            log_b_A[i,j] = self.model.log_transition_A_D + self.log_q_y(j+1) + log_b_D[i,j+1]
            log_b_I[i,j] = np.NINF
            log_b_D[i,j] = self.model.log_transition_D_D + self.log_q_y(j+1) + log_b_D[i,j+1]

        ####################################
        # Recursion
        ####################################                    
        for i in range( self.n-1, 0, -1 ):
            for j in range( self.m-1, 0, -1 ):
                A_end = self.log_p(i+1,j+1) + log_b_A[i+1,j+1]
                I_end = self.log_q_x(i+1)   + log_b_I[i+1,j]
                D_end = self.log_q_y(j+1)   + log_b_D[i,j+1]
            
                ### Probability of latter alignment if in state A at (i,j)
                log_b_A[i,j] = logsumexp_3(
                    self.model.log_transition_A_A + A_end,
                    self.model.log_transition_A_I + I_end,
                    self.model.log_transition_A_D + D_end,
                    )

                ### Probability of latter alignment if in state I (Insertion) at (i,j)
                log_b_I[i,j] = logsumexp_2(
                    self.model.log_transition_I_A + A_end,
                    self.model.log_transition_I_I + I_end,
                    )

                ### Probability of latter alignment if in state D at (i,j)
                log_b_D[i,j] = logsumexp_2(
                    self.model.log_transition_D_A + A_end,
                    self.model.log_transition_D_D + D_end,
                    )
        
        self.log_b_A = log_b_A        
        self.log_b_I = log_b_I
        self.log_b_D = log_b_D
        
        
class BaumWelch():

    def __init__(self, initial_p_match, initial_p_gap_start, initial_p_gap_extend, p_end, alphabet_size, pseudocount_match = 0.0, pseudocount_mismatch = 0.0, pseudocount_align = 0.0, pseudocount_gap_start = 0.0, pseudocount_gap_end = 0.0, pseudocount_gap_extend = 0.0):
        self.current_p_match = initial_p_match
        self.current_p_gap_start = initial_p_gap_start
        self.current_p_gap_extend = initial_p_gap_extend
        self.p_end = p_end
        self.alphabet_size = alphabet_size
        
        self.pseudocount_match = pseudocount_match
        self.pseudocount_mismatch = pseudocount_mismatch
        self.pseudocount_align = pseudocount_align        
        self.pseudocount_gap_start = pseudocount_gap_start
        self.pseudocount_gap_end = pseudocount_gap_end
        self.pseudocount_gap_extend = pseudocount_gap_extend
        
        
        
    def build_model(self):
        return Model( self.current_p_match, self.current_p_gap_start, self.current_p_gap_extend, self.p_end, self.alphabet_size )
        
    def iterate( self, data, steps = 10, delta = None ):
        prev_log_liklihood = np.NINF
        print( "Starting Baum-Welch on %d samples." % len(data) )
        for step in range(steps):
            model = self.build_model()
            
            self.current_p_match, self.current_p_gap_start, self.current_p_gap_extend = model.estimate_parameters( data, self.pseudocount_match, self.pseudocount_mismatch, self.pseudocount_align, self.pseudocount_gap_start, self.pseudocount_gap_end, self.pseudocount_gap_extend )
            log_likelihood = model.log_likelihood()
            print("Baum-Welch Interation:", step, log_likelihood, self.current_p_match, self.current_p_gap_start, self.current_p_gap_extend )
            
            if delta and abs( log_likelihood - prev_log_liklihood ) < delta:
                return model
                
        return model            
            
            
        
        
        
        
        

        
        
    



