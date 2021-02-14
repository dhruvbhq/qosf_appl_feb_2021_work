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
#from sympy.parsing.sympy_parser import parse_expr
from sympy import *
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
    def __init__(self, name, num_qubits, tolerance = 10**(-12)):
        super().__init__(name)
        assert(((np.ceil(num_qubits)) == np.floor(num_qubits) and (num_qubits > 0))), "Number of qubits must be a positive integer"
        self.num_qubits = num_qubits
        self.st_dim = 2**num_qubits
        # Statevector
        self.st_vec = np.zeros(self.st_dim) + np.zeros(self.st_dim)*1j
        self.tolerance = tolerance
        
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
            assert(np.abs(1 - np.linalg.norm(st_vec)) < self.tolerance), "Incorrect magnitude of the state vector"
            
        def init_state_SIQC_CUSTOM(self, st_vec):
            self.st_vec = st_vec
        
        def init_state_SIQC_GROUND(self, *args):
            ground_state = np.zeros(self.st_dim) + np.zeros(self.st_dim)*1j
            ground_state[0] = 1
            self.st_vec = ground_state
        
        def init_state_SIQC_RANDOM(self, *args):
            random_state = np.random.uniform(-1,1,self.st_dim) + np.random.uniform(-1,1,self.st_dim)*1j
            random_state = random_state/np.linalg.norm(random_state)
            self.st_vec = random_state
        
        init_method = "init_state_SIQC_" + init_type
        result = eval(init_method + "(self, st_vec)")
        assert(np.abs(1 - np.linalg.norm(self.st_vec)) < self.tolerance)
        
    # Quantum gates method
    def apply_gate(self, gate_type, target, control=None, parametric_gate=None, parameters=None):
        assert((gate_type == "H") or 
               (gate_type == "X") or 
               (gate_type == "Y") or 
               (gate_type == "Z") or 
               (gate_type == "I") or
               (gate_type == "S") or
               (gate_type == "T") or
               (gate_type == "PARAMETRIC")), "Unsupported Quantum Gate type."
        
        def return_1qubit_gate_matrix(self, gate_type, parametric_gate=None, parameters=None):
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
            # S gate
            S_1 = np.array([[1,0],[0,0]]) + (np.array([[0,0],[0,1]])*1j)
            # T gate
            T_1 = np.array([[1,0],[0,1/np.sqrt(2)]]) + (np.array([[0,0],[0,-1/np.sqrt(2)]])*1j)
            
            gate_map = {
                "H" : H_1,
                "X" : NOT_1,
                "Y" : ROT_Y_1,
                "Z" : PHASE_1,
                "I" : I_1,
                "S" : S_1,
                "T" : T_1}
            if(gate_type != "PARAMETRIC"):
                return gate_map[gate_type]
            else:
                #Parsing the parametric gate expression
                for i in range(2):
                    for j in range(2):
                        # Replace the parameters with their values
                        for parameter in parameters:                       
                            replace_by_me = str(parameters[parameter])
                            parametric_gate[i][j] = parametric_gate[i][j].replace(parameter, replace_by_me)
                        # And then evaluate the expression
                        parametric_gate[i][j] = parse_expr(parametric_gate[i][j])   
                        parametric_gate[i][j] = parametric_gate[i][j].evalf()   
                        parametric_gate[i][j] = complex(parametric_gate[i][j])
                return parametric_gate
        
        overall_gate_matrix = 1     
        # Endianness is little endian, ie |q3 q2 q1 q0>
        if(control == None):
            for i in range(self.num_qubits):
                if(i == target):
                    overall_gate_matrix = np.kron(
                        return_1qubit_gate_matrix(self, gate_type, parametric_gate, parameters),
                        overall_gate_matrix                        
                        ) # TODO
                else:
                    overall_gate_matrix = np.kron(
                        return_1qubit_gate_matrix(self, gate_type="I"),
                        overall_gate_matrix
                        )   
        
        assert(np.shape(overall_gate_matrix) == (self.st_dim, self.st_dim)), "Gate dimensions is in error."               
        self.st_vec = np.matmul(overall_gate_matrix, self.st_vec)
        assert(np.abs(1 - np.linalg.norm(self.st_vec)) < self.tolerance), "State vector normalization error."
            
    # Circuit execution method
    def execute_ckt(self, program):
        # program is supposed to be an array of dictionaries containing the steps
        # to be executed in the circuit
        assert((len(program)) > 0), "Please provide a valid program."
        for step in program:
            if not "control" in step:
                step["control"] = None
            assert(("parametric_gate" in step) == ("parameters" in step))
            if not "parametric_gate" in step:
                step["parametric_gate"] = None
            if not "parameters" in step:
                step["parameters"] = None
                
            self.apply_gate(step["gate_type"], step["target"], step["control"], step["parametric_gate"], step["parameters"])
      
    # Circuit measurement method
    def measure_ckt(self, num_shots, reporting_type="COUNT"):
        assert((reporting_type == "COUNT") or (reporting_type == "PERCENT")), "Unsupported reporting type."
        # Convert state vector into vector of probabilities of measuring a basis state
        prob_vec = np.power(np.abs(self.st_vec), 2)
        assert(np.abs(1 - np.sum(prob_vec)) < self.tolerance), "PDF normalization error. pdf = "+str(prob_vec)
        # Convert the probability distribution into a cumulative distribution function
        cdf = prob_vec
        for i in range(1, np.shape(cdf)[0]):
            cdf[i] += cdf[i-1]
        assert(np.abs(1 - cdf[-1]) < self.tolerance), "CDF normalization error"        
        
        # Initialize the results dictionary
        self.results = {}
        for i in range(np.shape(cdf)[0]):
            key = self.get_bin(i, self.num_qubits)
            self.results[key] = 0
            
        for n in range(num_shots):
            r = np.random.uniform(0, 1)
            assigned = 0
            for i in range(np.shape(cdf)[0] - 1):
                if((r <= cdf[0]) and (assigned == 0)):
                    key = self.get_bin(0, self.num_qubits)
                    self.results[key] += 1
                    assigned = 1
                elif((r > cdf[i]) and (r <= cdf[i+1]) and (assigned == 0)):
                    key = self.get_bin(i+1, self.num_qubits)
                    self.results[key] += 1
                    assigned = 1
                
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
        
    def give_bloch_angles(self):
        # Returns the angles theta and phi (in radian) for the Bloch vector 
        # corresponding  to the current state (only if the state is a 1 qubit state)
        # Under development
        if(self.num_qubits == 1):
            r0 = np.abs(self.st_vec[0])
            angle0 = np.angle(self.st_vec[0])
            r1 = np.abs(self.st_vec[1])
            angle1 = np.angle(self.st_vec[1])
            theta = 2*np.arccos(r0)
            phi = angle1 - angle0
            return theta, phi

#------------------------------------------------
