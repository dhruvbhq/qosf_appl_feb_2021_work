# -*- coding: utf-8 -*-
"""
SIQCANDAR simulator.

Original Author: Dhruv Bhatnagar.
Motivation: QOSF February 2021 screening task.
"""

# Imports
#------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------

# Classes
#------------------------------------------------

class siqc_root:
    # Constructor
    def __init__(self, name):
        self.name = name
    
    def print_name(self):
        print('Name of this object is:', self.name)
        

class siqc_ckt(siqc_root):
    # Constructor
    def __init__(self, name, num_qubits):
        super().__init__(name)
        assert(((np.ceil(num_qubits)) == np.floor(num_qubits) and (num_qubits > 0))), "Number of qubits must be a positive integer"
        self.num_qubits = num_qubits
        self.st_dim = 2**num_qubits
        # Statevector
        self.st_vec = np.zeros(self.st_dim) + np.zeros(self.st_dim)*1j
        
    # Auxiliary function to give an n-bit binary expansion
    def get_bin(self, x, n=0):
        return format(x, 'b').zfill(n)
        
    # State initialization method
    def init_state(self, init_type, st_vec = np.zeros(0)):
        assert((init_type == "CUSTOM") or (init_type == "GROUND") or (init_type == "RANDOM")), "Unupported Initialization Type."
        if(init_type == "CUSTOM"):
            assert(np.size(st_vec) == self.st_dim), "Incorrect dimension of the state vector"
            assert((np.log2(np.size(st_vec)) > 0)), "Incorrect dimension of the state vector."
            assert((np.ceil(np.log2(np.size(st_vec))) == (np.floor(np.log2(np.size(st_vec)))))), "Incorrect dimension of the state vector."
            assert(np.abs(1 - np.linalg.norm(np.abs(st_vec))) < 10**(-12)), "Incorrect magnitude of the state vector"
            
        def init_state_SIQC_CUSTOM(self, st_vec):
            self.st_vec = st_vec
        
        def init_state_SIQC_GROUND(self, *args):
            ground_state = np.zeros(self.st_dim) + np.zeros(self.st_dim)*1j
            ground_state[0] = 1
            self.st_vec = ground_state
        
        def init_state_SIQC_RANDOM(self, *args):
            random_state = np.random.uniform(-1,1,self.st_dim) + np.zeros(-1,1,self.st_dim)*1j
            random_state = random_state/np.linalg.norm(np.abs(random_state))
            self.st_vec = random_state
        
        init_method = "init_state_SIQC_" + init_type
        result = eval(init_method + "(self, st_vec)")
        assert(np.abs(1 - np.linalg.norm(np.abs(self.st_vec))) < 10**(-12))
        
    # Quantum gates method
    def apply_gate(self, gate, target, control=None):
        assert((gate == "H") or (gate == "X") or (gate == "Y") or (gate == "Z") or (gate == "I")), "Unsupported Quantum Gate type."
        
        # Single qubit gates
        # Identity
        I_1 = np.array([[1, 0], [0, 1]])
        # Hadamard
        H_1 = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)]])
        # Pauli X
        NOT_1 = np.array([[0, 1], [1, 0]])
        # Pauli Z
        PHASE_1 = np.array([[1, 0], [0, -1]])
        # Pauli Y
        ROT_Y_1 = np.array([[0, -1], [1, 0]])*1j
        
        gate_map = {
            "H" : H_1,
            "X" : NOT_1,
            "Y" : ROT_Y_1,
            "Z" : PHASE_1,
            "I" : I_1}
        
        overall_gate_matrix = 1        
        if(control == None):
            for i in range(self.num_qubits):
                if(i == target):
                    overall_gate_matrix = np.kron(overall_gate_matrix, gate_map[gate]) # TODO
                else:
                    overall_gate_matrix = np.kron(overall_gate_matrix, I_1)
                    
            assert(np.shape(overall_gate_matrix) == (self.st_dim, self.st_dim))
            
        self.st_vec = np.matmul(overall_gate_matrix, self.st_vec)
        assert(np.abs(1 - np.linalg.norm(np.abs(self.st_vec))) < 10**(-12)), "State vector normalization error."
            
    # Circuit execution method
    def execute_ckt(self, program):
        # program is supposed to be an array of dictionaries containing the steps
        # to be executed in the circuit
        assert((len(program)) > 0), "Please provide a valid program."
        for step in program:
            if not "control" in step:
                step["control"] = None
                
            self.apply_gate(step["gate"], step["target"], step["control"])
      
    # Circuit measurement method
    def measure_ckt(self, num_shots, reporting_type="COUNT"):
        assert((reporting_type == "COUNT") or (reporting_type == "PERCENT")), "Unsupported reporting type."
        # Convert state vector into vector of probabilities of measuring a basis state
        prob_vec = np.power(np.abs(self.st_vec), 2)
        assert(np.abs(1 - np.sum(prob_vec)) < 10**(-12)), "PDF normalization error."
        # Convert the probability distribution into a cumulative distribution function
        cdf = prob_vec
        for i in range(1, np.shape(cdf)[0]):
            cdf[i] += cdf[i-1]
        assert(np.abs(1 - cdf[-1]) < 10**(-12)), "CDF normalization error"
        
        
        # Initialize the results dictionary
        self.results = {}
        for i in range(np.shape(cdf)[0]):
            key = self.get_bin(i, self.num_qubits)
            self.results[key] = 0
            
        for n in range(num_shots):
            r = np.random.uniform(0, 1)
            incremented = 0
            for i in range(np.shape(cdf)[0]):
                key = self.get_bin(i, self.num_qubits)
                if((r >= cdf[i]) and (r < cdf[i+1])):
                    self.results[key] += 1
                    incremented = 1
            if(incremented == 0):
                self.results[self.get_bin(np.shape(cdf)[0]-1, self.num_qubits)] += 1
                
        if(reporting_type == "PERCENT"):
            for key in self.results:
                self.results[key] = self.results[key] * 100 / num_shots
                
        print("Results (", reporting_type, "):", self.results)
        
    def display_plot(self):
        plt.bar(list(self.results.keys()), self.results.values())
        plt.xlabel("Basis states")
        plt.ylabel("Measurement frequency/percent")
        plt.title("Plot of measurement outcomes")
        plt.show()
        
    def visualize_state(self):
        color = []
        indices = []
        st_vec_amp = np.power(np.abs(self.st_vec), 2)
        st_vec_phase = np.angle(self.st_vec)
        st_vec_phase = st_vec_phase / (2 * np.pi)
        for i in st_vec_phase:
            if(i > 0):
                color.append((0, 0, 1, i))
            elif(i < 0):
                color.append((1, 0, 0, np.abs(i)))
            else:
                color.append((0, 0, 0, 1))
                
        for i in range(self.st_dim):
            key = self.get_bin(i, self.num_qubits)
            indices.append(key)
        plt.bar(indices, st_vec_amp, color=color)
        plt.title("Basis state amplitudes and their phases.")
        plt.show()    
        print("Height is equal to magnitude of the probability amplitude. Phase is depicted by colour. Blue means a positive phase; red stands for negative phase and black colour depicts a real number. A higher red/blue colour intensity means a higher absolute phase angle.")

#------------------------------------------------