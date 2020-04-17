import numpy as np
from scipy.special import logsumexp

class Model():
    def __init__(self, p_mismatch, p_gap_start, p_gap_extend, p_end, alphabet_size):
        self.constant_log_q = np.log( 1.0/alphabet_size )
        self.alphabet_size = alphabet_size
        self.log_p_mismatch = np.log( p_mismatch )
        self.log_p_match = np.log( 1.0 - p_mismatch * (alphabet_size - 1) )
        
        self.log_transition_A_A = np.log( 1.0 - 2.0*p_gap_start - gap_end )
        self.log_transition_I_A = self.log_transition_D_A = np.log( 1.0 - p_gap_extend - gap_end )
        
        self.log_transition_A_I = self.log_transition_A_D = np.log( p_gap_start )
        self.log_transition_I_I = self.log_transition_D_D = np.log( p_gap_extend )
        
        self.log_transition_A_E = self.log_transition_I_E = self.log_transition_D_E = np.log( p_end )

    def log_q( self, item ):
        return self.constant_log_q
        
    def log_p( self, itemX, itemY ):
        return self.log_p_match if itemX == itemY else self.log_p_mismatch


        
    def estimate_parameters( self, data ):
        self.expected_values(data)
        
        estimated_p_mismatch = expected_mismatches/(expected_matches + expected_mismatches)
        estimated_p_gap_start = 0.5*(self.current_expected_A_I+self.current_expected_A_D)/(self.current_expected_A_A+current_expected_A_I+self.current_expected_A_D)
        estimated_p_gap_extend = 0.5*(self.current_expected_I_I + self.current_expected_D_D)/(
            self.current_expected_I_I + self.current_expected_D_D + self.current_expected_I_A + self.current_expected_D_A)
        
        raise estimated_p_mismatch, estimated_p_gap_start, estimated_p_gap_extend

    def expected_values( self, data ):    
        self.current_data = data
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
        
        for sequenceX,sequenceY in data:
            pair = SequencePair( self, sequenceX, sequenceY )
            log_probability_sequence = pair.forward_algorithm()
            pair.backward_algorithm()
            
            self.current_log_likelihood += log_probability_sequence
            
            for i in range( 1, self.n+1 ):
                for j in range( 1, self.m+1 ):
                    # Transition Expectations
                    self.current_expected_A_A += np.exp( pair.log_f_A[i,j] + self.log_transition_A_A + pair.log_p(i+1,j+1) + pair.b_A[i+1,j+1] - log_probability_sequence )
                    self.current_expected_A_I += np.exp( pair.log_f_A[i,j] + self.log_transition_A_I + pair.log_q_x( i+1 ) + pair.b_I[i+1,j]   - log_probability_sequence )
                    self.current_expected_A_D += np.exp( pair.log_f_A[i,j] + self.log_transition_A_D + pair.log_q_y( j+1 ) + pair.b_D[i,j+1]   - log_probability_sequence )
        
                    self.current_expected_I_A += np.exp( pair.log_f_I[i,j] + self.log_transition_I_A + pair.log_p(i+1,j+1) + pair.b_A[i+1,j+1] - log_probability_sequence )     
                    self.current_expected_I_I += np.exp( pair.log_f_I[i,j] + self.log_transition_I_I + pair.log_q_x( i+1 ) + pair.b_I[i+1,j]   - log_probability_sequence )     

                    self.current_expected_D_A += np.exp( pair.log_f_D[i,j] + self.log_transition_D_A + pair.log_p(i+1,j+1) + pair.b_A[i+1,j+1] - log_probability_sequence )        
                    self.current_expected_D_D += np.exp( pair.log_f_D[i,j] + self.log_transition_D_D + pair.log_q_y( j+1 ) + pair.b_D[i+1,j]   - log_probability_sequence )

                    # Emission Expectations
                    posterior_state_A = np.exp( pair.log_f_A[i,j] + pair.log_b_A[i,j] - log_probability_sequence )
                    if self.sequence_item_X(i) == self.sequence_item_Y(j)
                        self.current_expected_matches += posterior_state_A
                    else:
                        self.current_expected_mismatches += posterior_state_A
                    

class SequencePair():
    def __init__(self, model, sequenceX, sequenceY):
        self.model = model
        self.sequenceX = sequenceX
        self.sequenceY = sequenceY

        self.n = len(sequence1)
        self.m = len(sequence2)
        
        self.log_f_A = None
        self.log_f_I = None
        self.log_f_D = None

        self.log_f_E = None
        
        self.log_b_A = None
        self.log_b_I = None
        self.log_b_D = None
        
        
    def log_q_x( self, i ):
        if i >= self.n:
            return np.NINF
        return self.model.log_q( self.sequence_item_X(i) )
    def log_q_y( self, j ):
        if j >= self.m:
            return np.NINF
    
        return self.model.log_q( self.sequence_item_Y(j) )
        
    def sequence_item_X(self, i):
        return self.sequenceX[ i-1 ]
    def sequence_item_Y(self, j):
        return self.sequenceY[ j-1 ]
    
    def log_p( self, i, j ):
        if i >= self.n or j >= self.m:
            return np.NINF
    
        return self.model.log_p( self.sequence_item_X(i), self.sequence_item_Y(j) )

    def forward_algorithm( self ):
        '''
        Calculates the combined probability of all alignments up to position (i,j) that end in a particular state.
        R. Durbin, S. Eddy, A. Krogh, G. Mitchison, Biological Sequence Analysis, 87.
        
        Returns the log probability of the whole sequence pair log(P(x,y)) i.e. log(f_E(n,m))
        '''
        
        # Check if this function has already been called for this sequence pair
        if self.log_f_E:
            return self.log_f_E
            
        log_f_A = np.zeros( (self.n+1, self.m+1), dtype=np.float32 )
        log_f_I = np.zeros( (self.n+1, self.m+1), dtype=np.float32 )        
        log_f_D = np.zeros( (self.n+1, self.m+1), dtype=np.float32 )
        
        ####################################
        # Initial values at start boundaries
        ####################################
        # f_A( 0,0 ) has a probability of 1 so the log is 0.0 which is already initialized
        log_f_A[1:,0] = log_f_A[0,1:] = np.NINF
        
        # Start boundaries for insertion and deletion states
        log_f_I[0,0] = log_f_D[0,0] = np.NINF
        for i in range( self.n ):
            log_f_I[i+1,0] = self.log_q_x( i+1 ) + self.model.log_epsilon + log_f_I[i,0]
        for j in range( self.m ):
            log_f_D[0,j+1] = self.log_q_y( j+1 ) + self.model.log_epsilon + log_f_D[0,j]
            
        
        ####################################
        # Recursion
        ####################################
        for i in range( 1, self.n+1 ):
            for j in range( 1, self.m+1 ):

                ### Probability ends in Alignment
                log_f_A[i,j] = self.log_p( i, j ) + logsumexp( [
                    self.model.log_transition_A_A + log_f_A[i-1,j-1],
                    self.model.log_transition_I_A + log_f_I[i-1,j-1],
                    self.model.log_transition_D_A + log_f_D[i-1,j-1],
                    ] )
                    
                ### Probability ends in Insertion
                log_f_I[i,j] = self.log_q_x( i ) + logsumexp( [
                    self.model.log_transition_A_I + log_f_A[i-1,j],
                    self.model.log_transition_I_I + log_f_I[i-1,j],
                    ] )                

                ### Probability ends in Deletion
                log_f_D[i,j] = self.log_q_y( j ) + logsumexp( [
                    self.model.log_transition_A_D + log_f_A[i,j-1],
                    self.model.log_transition_D_D + log_f_D[i,j-1],
                    ] )
        
        ####################################
        # Termination
        ####################################
        self.log_f_E = logsumexp( [
                    self.model.log_transition_A_E + log_f_A[self.n, self.m],
                    self.model.log_transition_I_E + log_f_I[self.n, self.m],
                    self.model.log_transition_D_E + log_f_D[self.n, self.m],
                    ] )
        
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
        if self.log_b_A and self.log_b_D and self.log_b_D:
            return
        
        log_b_A = np.zeros( (self.n+2, self.m+2), dtype=np.float32 )        # The size is n+2,m+2 to make the recursion simpler, this should be optimized
        log_b_I = np.zeros( (self.n+2, self.m+2), dtype=np.float32 )        # The size is n+2,m+2 to make the recursion simpler, this should be optimized       
        log_b_D = np.zeros( (self.n+2, self.m+2), dtype=np.float32 )        # The size is n+2,m+2 to make the recursion simpler, this should be optimized
        
        ####################################
        # Initial values at start boundaries
        ####################################        
        log_b_A[self.n,:self.m] = np.NINF
        log_b_A[:self.n,self.m] = np.NINF        
                
        # Start boundaries for insertion and deletion states
        log_b_A[self.n+1,:] = log_b_A[:,self.m+1] = np.NINF
        log_b_I[self.n+1,:] = log_b_I[:,self.m+1] = np.NINF
        log_b_I[self.n+1,:] = log_b_I[:,self.m+1] = np.NINF

        log_b_A[self.n, self.m] = self.model.log_transition_A_E
        log_b_I[self.n, self.m] = self.model.log_transition_I_E
        log_b_D[self.n, self.m] = self.model.log_transition_D_E
        
        
        for i in range( 1, self.n+1, -1 ):
            for j in range( 1, self.m+1, -1 ):
                ### Probability of latter alignment if in state A at (i,j)
                log_b_A[i,j] = logsumexp( [
                    self.model.log_transition_A_A + self.log_p(i+1,j+1) + log_b_A[i+1,j+1],
                    self.model.log_transition_A_I + self.log_q_x(i+1)   + log_b_I[i+1,j],
                    self.model.log_transition_A_D + self.log_q_y(j+1)   + log_b_D[i,j+1],
                    ] )

                ### Probability of latter alignment if in state I (Insertion) at (i,j)
                log_b_I[i,j] = logsumexp( [
                    self.model.log_transition_I_A + self.log_p(i+1,j+1) + log_b_A[i+1,j+1],
                    self.model.log_transition_I_I + self.log_q_x(i+1)   + log_b_I[i+1,j],
                    ] )

                ### Probability of latter alignment if in state D at (i,j)
                log_b_D[i,j] = logsumexp( [
                    self.model.log_transition_D_A + self.log_p(i+1,j+1) + log_b_A[i+1,j+1],
                    self.model.log_transition_D_D + self.log_q_y(j+1)   + log_b_D[i,j+1],
                    ] )
        
        self.log_b_A = log_b_A        
        self.log_b_I = log_b_I
        self.log_b_D = log_b_D
        
        
class BaumWelch():
    def __init__(self, initial_p_mismatch, initial_p_gap_start, initial_p_gap_extend, p_end, alphabet_size):
        self.current_p_mismatch = initial_p_mismatch
        self.current_p_gap_start = initial_p_gap_start
        self.current_p_gap_extend = initial_p_gap_extend
        self.p_end = p_end
        self.alphabet_size = alphabet_size
        
    def build_model(self):
        return Model( self.current_p_mismatch, self.current_p_gap_start, self.current_p_gap_extend, self.p_end, self.alphabet_size )
        
    def iterate( self, data, steps = 10, delta = None ):
        prev_log_liklihood = np.NINF
        for step in steps:
            model = self.build_model()
            
            log_likelihood, self.current_p_mismatch, self.current_p_gap_start, self.current_p_gap_extend = model.estimate_parameters( data )
            
            if delta and abs( log_likelihood - prev_log_liklihood ) < delta:
                return model
                
        return model            
            
            
        
        
        
        
        

        
        
    



