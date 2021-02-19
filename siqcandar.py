# -*- coding: utf-8 -*-
"""
SIQCANDAR - Simulator of Quantum Computations for Amateurs, Novices, Developers
and Researchers.

Software to perform simulations of quantum computations.

Original Author: Dhruv Bhatnagar.
Motivation: QOSF February 2021 screening task.
"""

# Imports
#------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import turtle as tl
#------------------------------------------------

# Classes
#------------------------------------------------

class siqc_root:
    """
    A class used to represent the base class for everything in this simulator.
    The intention is to extend constructs like the quantum circuit from this    
    class. It can contain common attributes and methods.
    
    Attributes
    ----------
    name : str
        The name of the instance of this construct.
        
    Methods
    ----------
    print_name()
        Prints the name of this construct.
    """
    
    # Constructor
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            The name of the instance of this construct.
        """
        self.name = name
    
    def print_name(self):
        """
            Prints the name of this construct.
        """
        print('Name of this object is:', self.name)
        

class siqc_ckt(siqc_root):
    """
    A class used to simulate the behaviour and operations associated with a 
    quantum circuit.
    
    Attributes
    ----------
    num_qubits : int
        The number of qubits in this quantum circuit.
    st_dim : int
        The dimemsion of the statevector for this quantum circuit.
    st_vec : numpy array
        The statevector of the current state of this circuit.
    tolerance : float
        Used as an internal threshold for error reporting or assertion statements.
    st_hist : list
        The tracked history of states of this quantum circuit, resulting from 
        initialization and gates.
    op_hist : list
        The tracked history of operations performed on this quantum circuit.
    results : list
        The results obtained after multi-shot measurement of all qubits.
    __xloc : int
        Private variable used for circuit drawing.
        
    Methods
    ----------
    get_bin(x, n=0)
        Computes the n-bit binary representation of x
    init_state(init_type, st_vec = np.zeros(0))
        Used to initialize the quantum state.
    apply_gate(gate_type, target, control=None, parametric_gate=None, parameters=None)
        Used to apply a single quantum gate to the circuit. 
        Ideally the user should use execute_ckt in place of this.
    execute_ckt(program)
        Executes the given program on this quantum circuit. The program is a
        list of dictionaries, each representing a single gate operations. Each 
        dictionary can be used to specify the gate type, target/control qubits,
        and parametrized gates.
    measure_ckt(num_shots, reporting_type="COUNT")
        Performs multi-shot measurement of all the qubits and prints the 
        results. The results can be printed as actual counts or percentage, as
        specified by the user.
    plot_measure_results()
        Plots a bar diagram depicting the measurement results.
    visualize_state()
        Plots a visualization of the statevector depicting the probabilty 
        amplitudes and phases.
    give_bloch_angles()
        Returns the angles theta and phi (in radian) for the Bloch vector 
        corresponding  to the current state (only if the state is a 1 qubit 
        state) (Under development)
    give_state_by_label(label)
        Auxiiary method which only returns the state vector corresponding to a
        quantum state specified by a binary string as the label. eg |1010>
    draw_ckt()
        Method used to draw the contructed quantum circuit using turtle.
    """
    # Constructor
    def __init__(self, name, num_qubits, tolerance = 10**(-12)):
        """
        Parameters
        ----------
        name : int
            The name of this quantum circuit.
        num_qubits : int
            The number of qubits in this quantum circuit.
        tolerance : float, optional
            Used as an internal threshold for error reporting or assertion 
            statements. The default is 10**(-12).

        Returns
        -------
        None.

        """
        super().__init__(name)
        assert(((np.ceil(num_qubits)) == np.floor(num_qubits) and 
                (num_qubits > 0))), "Number of qubits must be a positive integer"
        self.num_qubits = num_qubits
        self.st_dim = 2**num_qubits
        # Statevector
        self.st_vec = np.zeros(self.st_dim) + np.zeros(self.st_dim)*1j
        self.tolerance = tolerance
        self.st_hist = []
        self.op_hist = []
        self.results = None
        self.__xloc = 0
        
    # Auxiliary function to give an n-bit binary representation
    def get_bin(self, x, n=0):
        """
        Computes the n-bit binary representation of x

        Parameters
        ----------
        x : int
            The number for which the binary representation is required.
        n : int, optional
            The number of bits used to represent the required binary 
            representation. The default is 0, in which case only those many 
            bits as required will be used.

        Returns
        -------
        str
            The n-bit binary representation of x as a string.

        """
        return format(x, 'b').zfill(n)
        
    # State initialization method
    def init_state(self, init_type, st_vec = np.zeros(0)):
        """
        Used to initialize the quantum state.

        Parameters
        ----------
        init_type : str
            The type of initialization. The following choices are currently 
            supported:
                "GROUND"
                    Initializes the circuit in the ground state.
                "CUSTOM"
                    Initializes the circuit to the state provided by the 
                    argument st_vec
                "RANDOM"
                    Initializes the ciruit's state to a random complex vector 
                    with unit norm.
        st_vec : numpy array, optional
            The state vector to which the circuit has to be initialized in case 
            of "CUSTOM" init_type. The default is np.zeros(0).

        Returns
        -------
        None.

        """
        assert((init_type == "CUSTOM") or 
               (init_type == "GROUND") or 
               (init_type == "RANDOM")), "Unupported Initialization Type."
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
        self.st_hist.append(self.st_vec)
        self.op_hist.append({"initialize" : ""})
        assert(np.abs(1 - np.linalg.norm(self.st_vec)) < self.tolerance)
        
    # Quantum gates method
    def apply_gate(self, gate_type, target, control=None, parametric_gate=None, parameters=None):
        """
        Used to apply a single quantum gate to the circuit. 
        Ideally the user should use execute_ckt in place of this. Note that 
        the endianness convention is little endian, ie |q3 q2 q1 q0>.

        Parameters
        ----------
        gate_type : str
            Specifies the (single target) quantum gate type. Currently 
            supported types are:
                "H"
                    Hadamard gate.
                "X"
                    Pauli X gate (Not).
                "Y"
                    Pauli Y gate.
                "Z"
                    Pauli Z gate (phase flip).
                "I"
                    Identity gate.
                "S"
                    S gate.
                "T"
                    T gate.
                "PARAMETRIC"
                    Parametrized gate for variational circuits. With this, user
                    needs to specify the gate matrix as a list with functional 
                    expressions as strings, and the involved parameters as a 
                    dictionary.
        target : int
            The (single) target qubit.
        control : int, optional
            The control qubit if the gate is a controlled unitary. Currently, 
            only those gates controlled by a single qubit are supported. The
            default is None.
        parametric_gate : list, optional
            The list specifying the matrix elements of a parametrized gate, if 
            the gate type is parametric. This method is capable of parsing the 
            string expressions containing parameters to mathematical functions 
            with the specfied parameter values. The default is None.
        parameters : dictionary, optional
            Specifes the parameter values for the parametric gate in a 
            dictionary. The default is None.

        Returns
        -------
        None.

        """
        assert((gate_type == "H") or 
               (gate_type == "X") or 
               (gate_type == "Y") or 
               (gate_type == "Z") or 
               (gate_type == "I") or
               (gate_type == "S") or
               (gate_type == "T") or
               (gate_type == "PARAMETRIC")), "Unsupported Quantum Gate type."
        
        def return_1qubit_gate_matrix(self, gate_type, parametric_gate=None, parameters=None):
            """
            Returns the matrix for a single qubit unitary gate, which may be 
            parametrized.

            Parameters
            ----------
            gate_type : str
                See documentation of apply_gate.
            parametric_gate : list, optional
                See documentation of apply_gate. The default is None.
            parameters : dictionary, optional
                See documentation of apply_gate. The default is None.

            Returns
            -------
            numpy array
                The matrix corresponding to the required single qubit gate.

            """
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
            #Generate operator matrix for the whole circuit by repeated
            #Kronecker product
            for i in range(self.num_qubits):
                if(i == target):
                    overall_gate_matrix = np.kron(
                        return_1qubit_gate_matrix(self, gate_type, parametric_gate, parameters),
                        overall_gate_matrix)
                else:
                    overall_gate_matrix = np.kron(
                        return_1qubit_gate_matrix(self, gate_type="I"),
                        overall_gate_matrix)
        else:
            #Controlled gate case
            #Forms the gate expression according to the reference material.
            #Currently, it only supports single qubit as control and single 
            #qubit as the target.
            temp_gate = np.zeros([self.st_dim, self.st_dim]) + np.zeros([self.st_dim, self.st_dim])*1j
            #if the partial term in the matrix is of the form 
            #I kron ... kron (control_qubit) kron I ... kron (target_op) kron I ... kron I
            #then, denote them as  head  | control_qubit | mid | target_op | trail
            #control and target can be each others' positions as well; assume that this connotation still holds
            head_size = self.num_qubits - 1 - max(control, target)
            mid_size = abs(control - target) - 1
            trail_size = min(control, target)
            #k - index over control states: assuming that control qubit is a single qubit
            for k in range(2):
                overall_gate_matrix = 1
                mid = 1
                head = 1
                #appending the trail first
                for t in range(trail_size):
                    overall_gate_matrix = np.kron(
                        return_1qubit_gate_matrix(self, gate_type="I"),
                        overall_gate_matrix)
                #form the mid beforehand
                for m in range(mid_size):
                    mid = np.kron(
                        return_1qubit_gate_matrix(self, gate_type="I"),
                        mid)
                #forming the head beforehand
                for h in range(head_size):
                    head = np.kron(
                        return_1qubit_gate_matrix(self, gate_type="I"),
                        head)
                if(k == 1):
                    #Control is |1>
                    if(target < control):
                        #append actual single qubit unitary
                        overall_gate_matrix = np.kron(
                            return_1qubit_gate_matrix(self, gate_type, parametric_gate, parameters),
                            overall_gate_matrix)
                    else:
                        #append state of control qubit |1><1|
                        overall_gate_matrix = np.kron(
                            np.outer([0,1],[0,1]),
                            overall_gate_matrix)
                    #append the mid section
                    overall_gate_matrix = np.kron(mid, overall_gate_matrix)
                    #append the remaining control or target operator
                    if(target < control):
                        #append state of control qubit |1><1|
                        overall_gate_matrix = np.kron(
                            np.outer([0,1],[0,1]),
                            overall_gate_matrix)                            
                    else:
                        #append actual single qubit unitary
                        overall_gate_matrix = np.kron(
                            return_1qubit_gate_matrix(self, gate_type, parametric_gate, parameters),
                            overall_gate_matrix)
                else:
                    #Control is |0>.
                    if(target < control):
                        #append identity
                        overall_gate_matrix = np.kron(
                            return_1qubit_gate_matrix(self, gate_type="I"),
                            overall_gate_matrix)
                    else:
                        #append state of control qubit |0><0|
                        overall_gate_matrix = np.kron(
                            np.outer([1,0],[1,0]),
                            overall_gate_matrix)
                    #append the mid section
                    overall_gate_matrix = np.kron(mid, overall_gate_matrix)
                    #append the remaining control or target operator
                    if(target < control):
                        #append state of control qubit |0><0|
                        overall_gate_matrix = np.kron(
                            np.outer([1,0],[1,0]),
                            overall_gate_matrix)                            
                    else:
                        #append identity
                        overall_gate_matrix = np.kron(
                            return_1qubit_gate_matrix(self, gate_type = "I"),
                            overall_gate_matrix)
                #Finally, add the head
                overall_gate_matrix = np.kron(head, overall_gate_matrix)
                temp_gate += overall_gate_matrix
            overall_gate_matrix = temp_gate
        
        assert(np.shape(overall_gate_matrix) == (self.st_dim, self.st_dim)), "Gate dimensions is in error."               
        self.st_vec = np.matmul(overall_gate_matrix, self.st_vec)        
        assert(np.abs(1 - np.linalg.norm(self.st_vec)) < self.tolerance), "State vector normalization error."
            
    # Circuit execution method
    def execute_ckt(self, program):
        """
        Executes the given program on this quantum circuit. The program is a
        list of dictionaries, each representing a single gate operations. Each 
        dictionary can be used to specify the gate type, target/control qubits,
        and parametrized gates.

        Parameters
        ----------
        program : list of dictionaries
            program is supposed to be a list of dictionaries containing the 
            steps to be executed in the circuit. See the documentation for 
            apply_gate. Each element of this list should represent a single 
            gate operation, which may be a standard gate, parametrized or a 
            controlled gate. Each dictionary should contain the relevant key, 
            value pairs, with the keys/values being:
                "gate_type": 
                    one of "H", "X", "Y", "Z", "I", "S", "T", "PARAMETRIC"
                "target" : int
                    The (single) target qubit.
                "control" : int, optional
                    The control qubit if the gate is a controlled unitary. 
                    Currently, only those gates controlled by a single qubit 
                    are supported. The default is None.
                "parametric_gate" : list, optional
                    The list specifying the matrix elements of a parametrized 
                    gate, if the gate type is parametric. This method is capable 
                    of parsing the string expressions containing parameters to 
                    mathematical functions with the specfied parameter values. 
                    The default is None.
                "parameters" : dictionary, optional
                Specifes the parameter values for the parametric gate in a
                dictionary. The default is None.

        Returns
        -------
        None.

        """        
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
            self.st_hist.append(self.st_vec)
            self.op_hist.append(step)
      
    # Circuit measurement method
    def measure_ckt(self, num_shots, reporting_type="COUNT", quiet=False):
        """
        Performs multi-shot measurement of all the qubits and prints the 
        results. Implements a logic for weighted random sampling from an 
        arbitrary probability distribution. The results can be printed as actual
        counts or percentage, as specified by the user.        

        Parameters
        ----------
        num_shots : int
            The number of shpts or trials to carry out the measurement.
        reporting_type : str, optional
            Specifies whether to report the results as "COUNT" or "PERCENT".
            The default is "COUNT".
        quiet : Bool, optional
            Specifies whether to print the measurement results or not. 
            Regardless of this argument, they are stored in self.results

        Returns
        -------
        None.

        """
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
        if(quiet == False):        
            print("Results (", reporting_type, "):", self.results)
        self.op_hist.append({"measure" : ""})
        
    def plot_measure_results(self):
        """
        Plots a bar diagram depicting the measurement results.

        Returns
        -------
        None.

        """    
        plt.bar(list(self.results.keys()), self.results.values())
        plt.xlabel("Basis states")
        plt.ylabel("Measurement frequency/percent")
        plt.title("Plot of measurement outcomes")
        plt.show()
        
    def visualize_state(self):
        """
        Plots a visualization of the statevector depicting the probabilty 
        amplitudes and phases. Height is equal to magnitude of the probability
        amplitude. Phase is depicted by colour. Blue means a positive phase;
        red stands for negative phase and black colour depicts a real number 
        (ie phase is zero). A higher red/blue colour intensity means a higher 
        absolute phase angle.

        Returns
        -------
        None.

        """        
    
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
        plt.xticks(rotation=45)
        plt.show()    
        print("Height is equal to magnitude of the probability amplitude.",
              "Phase is depicted by colour. Blue means a positive phase;",
              "red stands for negative phase and black colour depicts a real number.",
              "A higher red/blue colour intensity means a higher absolute phase angle.")
        
    def give_bloch_angles(self):
        """
        Returns the angles theta and phi (in radian) for the Bloch vector 
        corresponding  to the current state (only if the state is a 1 qubit 
        state) (Under development)

        Returns
        -------
        theta : float
            angle theta corresponding to the Bloch representation.
        phi : float
            angle phi corresponding to the Bloch representation.

        """        
    
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
        
    def give_state_by_label(self, label):
        """
        Auxiiary method which only returns the state vector corresponding to a
        quantum state specified by a binary string as the label. (eg to obtain 
        the state vector for |1010>, pass "1010" to this function as the 
        argument. This example assumes that number of qubits in the circuit is 
        atleast 4.)

        Parameters
        ----------
        label : str
            Binary representation of the label of the required state vector.

        Returns
        -------
        st_vec : numpy array
            The state vector corresponding to the state |label>.

        """
        one_hot_idx = int(label, 2)
        st_vec = np.zeros(self.st_dim)
        st_vec[one_hot_idx] = 1
        return st_vec  

    def draw_ckt(self):
        """
        Method used to draw the contructed quantum circuit using turtle.        

        Returns
        -------
        None.

        """

        self.__xloc = 0
        
        def draw_init(lc_op):
            tl.penup()
            self.__xloc = -200
            tl.setpos(x=self.__xloc, y=0)
            tl.pendown()
            #for qubit labels and initial wires
            for i in range(self.num_qubits):
                tl.forward(20)
                tl.backward(20)
                tl.penup()
                tl.backward(20)
                tl.pendown()
                tl.write(self.name + '_' + str(i))
                tl.penup()
                tl.forward(20)
                tl.pendown()
                if(i < self.num_qubits - 1):
                    tl.left(90)
                    tl.penup()
                    tl.forward(40)
                    tl.right(90)
                    tl.pendown()
            self.__xloc += 20
            tl.penup()
            tl.setpos(x=self.__xloc, y=0)
            #Init block
            tl.pendown()
            tl.right(90)
            tl.forward(10)
            tl.backward(10)
            tl.left(180)
            
            for i in range(self.num_qubits):
                if(i < self.num_qubits - 1):
                    tl.forward(40)
                else:
                    tl.forward(10)
            tl.right(90)
            tl.forward(40)
            tl.right(90)
            tl.forward(10)
            for i in range(self.num_qubits):
                if(i < self.num_qubits - 1):
                    tl.forward(40)
                else:
                    tl.forward(10)
            tl.right(90)
            tl.forward(40)
            tl.right(180)
            self.__xloc += 15
            tl.penup()           
            tl.setpos(x=self.__xloc, y=((self.num_qubits-1)*20))
            tl.pendown()
            tl.write("INIT")
            tl.penup()            
            self.__xloc += 25
            tl.setpos(x=self.__xloc, y=0)
            tl.pendown()
            
        def draw_meas(lc_op):
            #for initial wires
            tl.pendown()
            for i in range(self.num_qubits):
                tl.forward(10)
                tl.backward(10)
                if(i < self.num_qubits - 1):
                    tl.left(90)
                    tl.penup()
                    tl.forward(40)
                    tl.right(90)
                    tl.pendown()
            self.__xloc += 10
            tl.penup()
            tl.setpos(x=self.__xloc, y=0)
            #Meas block
            tl.pendown()
            tl.right(90)
            tl.forward(10)
            tl.backward(10)
            tl.left(180)
            
            for i in range(self.num_qubits):
                if(i < self.num_qubits - 1):
                    tl.forward(40)
                else:
                    tl.forward(10)
            tl.right(90)
            tl.forward(40)
            tl.right(90)
            tl.forward(10)
            for i in range(self.num_qubits):
                if(i < self.num_qubits - 1):
                    tl.forward(40)
                else:
                    tl.forward(10)
            tl.right(90)
            tl.forward(40)
            tl.right(180)
            self.__xloc += 10
            tl.penup()           
            tl.setpos(x=self.__xloc, y=((self.num_qubits-1)*20))
            tl.pendown()
            tl.write("MEAS")
            tl.penup()
            tl.setpos(x=self.__xloc, y=((self.num_qubits-1)*20)+10)
            tl.pendown()
            tl.write("  =>")
            tl.penup()            
            self.__xloc += 30
            tl.setpos(x=self.__xloc, y=0)
            tl.pendown()
            
        def draw_gate(lc_op):
            #for initial lines
            tl.pendown()
            for i in range(self.num_qubits):
                tl.forward(10)
                tl.backward(10)
                if(i < self.num_qubits - 1):
                    tl.left(90)
                    tl.penup()
                    tl.forward(40)
                    tl.right(90)
                    tl.pendown()
            self.__xloc += 10
            tl.penup()
            tl.setpos(x=self.__xloc, y=0)
            tl.penup()            
            for i in range(self.num_qubits):
                if(i != lc_op["target"]):
                    #lines for other qubits
                    tl.setpos(x=self.__xloc, y=(i*40))
                    tl.pendown()
                    tl.forward(40)                    
                    tl.penup()
                else:
                    #box for gate
                    tl.setpos(x=self.__xloc, y=(lc_op["target"]*40))
                    tl.left(90)
                    tl.pendown()
                    tl.forward(20)
                    for i in range(3):
                        tl.right(90)
                        tl.forward(40)
                    tl.right(90)
                    tl.forward(20)
                    tl.right(90)
                    tl.penup()
                    tl.setpos(x=self.__xloc+20, y=(lc_op["target"]*40))
                    tl.pendown()
                    if(lc_op["gate_type"] != "PARAMETRIC"):
                        tl.write(lc_op["gate_type"])
                    else:
                        tl.write("U_p")
                    tl.penup()
            if(lc_op["control"] != None):
                tl.setpos(x=self.__xloc+20, y=(lc_op["control"]*40))
                tl.pendown()
                tl.dot(size=7)
                tl.penup()
                if(lc_op["control"] > lc_op["target"]):
                    tl.right(90)
                else:
                    tl.left(90)
                tl.pendown()
                tl.forward(20+((abs(lc_op["control"]-lc_op["target"])-1)*40))
                tl.penup()
                if(lc_op["control"] > lc_op["target"]):
                    tl.left(90)
                else:
                    tl.right(90)
                
            self.__xloc += 40
            tl.setpos(x=self.__xloc, y=0)
        
        for op in self.op_hist:
            if "initialize" in op:
                draw_init(op)
            elif "measure" in op:
                draw_meas(op)
            else:
                draw_gate(op)
                
        tl.hideturtle()
        tl.exitonclick()
#------------------------------------------------
