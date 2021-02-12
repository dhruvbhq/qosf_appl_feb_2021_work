# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:35:46 2021

@author: dbhatnag

This file contains the test scenarios for the SIQCANDAR simulator program.
"""

# Imports
#---------------------------------------------------
import numpy as np
import siqcandar
import time
#---------------------------------------------------

num_tests = 0
tic = 0

def launch_tests():
    # Endianness is little endian, ie |q3 q2 q1 q0>
    # Some useful defines
    ket0 = np.array([1, 0]) # |0>
    ket1 = np.array([0, 1]) # |1>
    
    ketp = (1/np.sqrt(2))*(ket0 + ket1) # |+>
    ketm = (1/np.sqrt(2))*(ket0 - ket1) # |->
    
    ket00 = np.kron(ket0, ket0)
    ket01 = np.kron(ket0, ket1)
    ket10 = np.kron(ket1, ket0)
    ket11 = np.kron(ket1, ket1)
    
    ket0p = np.kron(ket0, ketp)
    ket1p = np.kron(ket1, ketp)
    ketp0 = np.kron(ketp, ket0)
    ketp1 = np.kron(ketp, ket1)
    
    ket0m = np.kron(ket0, ketm)
    ket1m = np.kron(ket1, ketm)
    ketm0 = np.kron(ketm, ket0)
    ketm1 = np.kron(ketm, ket1)
    
    ket000p = np.kron(ket0, np.kron(ket0, np.kron(ket0, ketp)))
    ket00p0 = np.kron(ket0, np.kron(ket0, np.kron(ketp, ket0)))
    ket0p00 = np.kron(ket0, np.kron(ketp, np.kron(ket0, ket0)))
    ketp000 = np.kron(ketp, np.kron(ket0, np.kron(ket0, ket0)))
    
    ket00pp = np.kron(ket0, np.kron(ket0, np.kron(ketp, ketp)))
    ket0p0p = np.kron(ket0, np.kron(ketp, np.kron(ket0, ketp)))
    ketp00p = np.kron(ketp, np.kron(ket0, np.kron(ket0, ketp)))
    ket0pp0 = np.kron(ket0, np.kron(ketp, np.kron(ketp, ket0)))
    ketp0p0 = np.kron(ketp, np.kron(ket0, np.kron(ketp, ket0)))
    ketpp00 = np.kron(ketp, np.kron(ketp, np.kron(ket0, ket0)))
    
    ketpp = np.kron(ketp, ketp)
    ketpppp = np.kron(ketp, np.kron(ketp, np.kron(ketp, ketp)))
    
    def print_init_tests():
        global tic
        print("Running tests ...")
        print("---------------------------------------------------")
        tic = time.time()
        return
    
    def pre_test_task():
        global num_tests
        num_tests = num_tests + 1
        print("Running test ", num_tests, " -------------------------------------")
        return
    
    def print_report_tests():
         print("---------------------------------------------------")
         print("Tests passed. Number of tests run = ", num_tests, ". Time consumed = ", time.time() - tic, ". If you see this line without encountering any errors, it means the tests have passed.")
    
    print_init_tests()
    
    #Test 1	- State dimension - Quantum state dimension for 1 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    q1.init_state(init_type="GROUND")
    assert(q1.st_dim == 2**1), "Test 1 failed."
    
    #Test 2	- State dimension - Quantum state dimension for 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    assert(q1.st_dim == 2**2), "Test 2 failed."
    
    #Test 3	- State dimension - Quantum state dimension for 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    assert(q1.st_dim == 2**4), "Test 3 failed."
    
    #Test 4	- State initialization - Initialize 1 qubit state to user defined value with real probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    psi1 = np.array([1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 4 failed."
    
    #Test 5	- State initialization - Initialize 2 qubit state to user defined value with real probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    psi1 = np.array([1/np.sqrt(13), np.sqrt(2)/np.sqrt(13), -1*np.sqrt(3)/np.sqrt(13), np.sqrt(7)/np.sqrt(13)])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 5 failed."
    
    #Test 6	- State initialization - Initialize 4 qubit state to user defined value with real probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    psi1 = np.array([1/np.sqrt(25), -1*1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25),
                     1/np.sqrt(25), 1/np.sqrt(25), -1*1/np.sqrt(25), 1/np.sqrt(25),
                     1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25), -1*1/np.sqrt(25),
                     -1*1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25), np.sqrt(10)/np.sqrt(25)])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 6 failed."

    #Test 7	- State initialization - Initialize 1 qubit state to user defined value with complex probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1, tolerance=10**(-8))
    psi1 = np.array([1/np.sqrt(3), (np.sqrt(2)/np.sqrt(3))*1j])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 7 failed."
    
    #Test 8	- State initialization - Initialize 2 qubit state to user defined value with complex probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2, tolerance=10**(-8))
    psi1 = np.array([0.12212234-0.59549103j, -0.30416912+0.29570485j,
                     0.46135097-0.45942149j,  0.00583572-0.16300147j])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 8 failed."
    
    #Test 9	- State initialization - Initialize 4 qubit state to user defined value with complex probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4, tolerance=10**(-8))
    psi1 = np.array([0.21584501+0.2494106j ,  0.05495676+0.03468601j,
                     -0.05731732+0.15385203j,  0.1354276 -0.13732148j,
                     0.24636623-0.26832918j,  0.19739793+0.19480392j,
                     -0.04900181-0.02107175j,  0.11115504-0.21241653j,
                     0.24895018+0.26924763j,  0.11261091-0.15254312j,
                     0.04105645-0.22667174j, -0.21916044-0.28315859j,
                     -0.01293583+0.09801503j, -0.15619437-0.21974565j,
                     -0.25840702+0.15058432j, -0.06511822-0.15826991j])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 9 failed."
    
    #Test 10 - State initialization - Initialize 1 qubit state to ground state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    psi1 = np.zeros(2) + np.zeros(2)*1j
    psi1[0] = 1
    q1.init_state(init_type="GROUND")
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 10 failed."
    
    #Test 11 - State initialization - Initialize 2 qubit state to ground state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    psi1 = np.zeros(4) + np.zeros(4)*1j
    psi1[0] = 1
    q1.init_state(init_type="GROUND")
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 11 failed."
    
    #Test 12 - State initialization	- Initialize 4 qubit state to ground state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    psi1 = np.zeros(16) + np.zeros(16)*1j
    psi1[0] = 1
    q1.init_state(init_type="GROUND")
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 12 failed."
    
    #Test 13 - State initialization - Initialize 1 qubit state to random state and check norm
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    q1.init_state(init_type="RANDOM")
    psi1 = q1.st_vec
    assert(np.abs(1-np.linalg.norm(psi1)) < 10**(-8)), "Test 13 failed."
    q1.init_state(init_type="RANDOM")
    assert(np.abs(1-np.linalg.norm(q1.st_vec)) < 10**(-8)), "Test 13 failed."
    for i in range(len(psi1)):
        assert(psi1[i] != q1.st_vec[i]), "Test 13 failed."
    
    #Test 14 - State initialization	- Initialize 2 qubit state to random state and check norm
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="RANDOM")
    psi1 = q1.st_vec
    assert(np.abs(1-np.linalg.norm(psi1)) < 10**(-8)), "Test 14 failed."
    q1.init_state(init_type="RANDOM")
    assert(np.abs(1-np.linalg.norm(q1.st_vec)) < 10**(-8)), "Test 14 failed."
    for i in range(len(psi1)):
        assert(psi1[i] != q1.st_vec[i]), "Test 14 failed."
    
    #Test 15 - State initialization	- Initialize 4 qubit state to random state and check norm
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="RANDOM")
    psi1 = q1.st_vec
    assert(np.abs(1-np.linalg.norm(psi1)) < 10**(-8)), "Test 15 failed."
    q1.init_state(init_type="RANDOM")
    assert(np.abs(1-np.linalg.norm(q1.st_vec)) < 10**(-8)), "Test 15 failed."
    for i in range(len(psi1)):
        assert(psi1[i] != q1.st_vec[i]), "Test 15 failed."
    
    #Test 16 - Quantum gates and measurement - Apply H to 1 qubit state
    pre_test_task()
    #First, to the state |0>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    q1.init_state(init_type="GROUND")
    psi1 = (1/np.sqrt(2))*np.array([1, 1])
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 16 failed."
        
    q1.measure_ckt(num_shots=1000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0" : 500, "1" : 500}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 16 failed."
        
    #Next, to the state |1>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    psi_init = np.array([0, 1])
    psi1 = (1/np.sqrt(2))*np.array([1, -1])
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 16 failed."
        
    q1.measure_ckt(num_shots=1000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0" : 500, "1" : 500}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 16 failed."       
    
    #Test 17 - Quantum gates and measurement - Apply H to qubit 0 of 2 qubit state
    pre_test_task()
    #To the state |00>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    psi1 = ket0p
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 17 failed."
        
    q1.measure_ckt(num_shots=1000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 500, "01" : 500, "10" : 0, "11" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 17 failed."
        
    #Test 18 - Quantum gates and measurement - Apply H to qubit 1 of 2 qubit state
    pre_test_task()
    #To the state |01>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec = ket01)
    psi1 = ketp1
    program = [{"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 18 failed."
        
    q1.measure_ckt(num_shots=1000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 0, "01" : 500, "10" : 0, "11" : 500}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 18 failed."
    #Now to the state |10>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec = ket10)
    psi1 = ketm0
    program = [{"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 18 failed."
        
    q1.measure_ckt(num_shots=1000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 500, "01" : 0, "10" : 500, "11" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 18 failed."
        
    #Test 19 - Quantum gates and measurement -	Apply H to qubits 0 and 1 of 2 qubit state
    #To the state |00>
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    psi1 = ketpp
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 19 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 500, "01" : 500, "10" : 500, "11" : 500}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 19 failed."
        
    #Test 20 - Quantum gates and measurement - Apply H to qubit 0 of 4 qubit state
    #To the state |0000>
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ket000p
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 20 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 1000, 
                "0001" : 1000,
                "0010" : 0,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 0, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 20 failed."
        
    #Test 21 - Quantum gates and measurement - Apply H to qubit 1 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ket00p0
    program = [{"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 21 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 1000, 
                "0001" : 0,
                "0010" : 1000,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 0, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 21 failed."
        
    #Test 22 - Quantum gates and measurement - Apply H to qubit 2 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ket0p00
    program = [{"gate_type" : "H", "target" : 2}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 22 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 1000, 
                "0001" : 0,
                "0010" : 0,
                "0011" : 0,
                "0100" : 1000, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 0, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 22 failed."
        
    #Test 23 - Quantum gates and measurement - Apply H to qubit 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ketp000
    program = [{"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 23 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 1000, 
                "0001" : 0,
                "0010" : 0,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 1000, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 23 failed."
        
    #Test 24 - Quantum gates and measurement - Apply H to qubits 0 and 1 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ket00pp
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 24 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 500, 
                "0001" : 500,
                "0010" : 500,
                "0011" : 500,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 0, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 24 failed."
        
    #Test 25 - Quantum gates and measurement - Apply H to qubits 1 and 2 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ket0pp0
    program = [{"gate_type" : "H", "target" : 1}, {"gate_type" : "H", "target" : 2}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 25 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 500, 
                "0001" : 0,
                "0010" : 500,
                "0011" : 0,
                "0100" : 500, 
                "0101" : 0,
                "0110" : 500,
                "0111" : 0,
                "1000" : 0, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 25 failed."
        
    #Test 26 - Quantum gates and measurement - Apply H to qubits 2 and 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ketpp00
    program = [{"gate_type" : "H", "target" : 2}, {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 26 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 500, 
                "0001" : 0,
                "0010" : 0,
                "0011" : 0,
                "0100" : 500, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 500, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 500, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 26 failed."
        
    #Test 27 - Quantum gates and measurement - Apply H to qubits 0 and 2 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ket0p0p
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 2}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 27 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 500, 
                "0001" : 500,
                "0010" : 0,
                "0011" : 0,
                "0100" : 500, 
                "0101" : 500,
                "0110" : 0,
                "0111" : 0,
                "1000" : 0, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 27 failed."
        
    #Test 28 - Quantum gates and measurement - Apply H to qubits 1 and 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ketp0p0
    program = [{"gate_type" : "H", "target" : 1}, {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 28 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 500, 
                "0001" : 0,
                "0010" : 500,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 500, 
                "1001" : 0,
                "1010" : 500,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 28 failed."
        
    #Test 29 - Quantum gates and measurement - Apply H to qubits 0 and 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    psi1 = ketp00p
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(psi1)):
        assert(np.linalg.norm(psi1[i] - q1.st_vec[i]) < 10**(-8)), "Test 29 failed."
        
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 500, 
                "0001" : 500,
                "0010" : 0,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 500, 
                "1001" : 500,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 0.1*exp_prob[key]), "Test 29 failed."


    print_report_tests()
    return
