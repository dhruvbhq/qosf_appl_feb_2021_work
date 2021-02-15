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
    
    # defines the tolerance up to which measurements should be considered acceptable
    # by default, +/-10% of expected measurement frequency is acceptable
    meas_tol = 0.1
    q1 = None
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
    
    ket0ppp = np.kron(ket0, np.kron(ketp, np.kron(ketp, ketp)))
    ketp0pp = np.kron(ketp, np.kron(ket0, np.kron(ketp, ketp)))
    ketpp0p = np.kron(ketp, np.kron(ketp, np.kron(ket0, ketp)))
    ketppp0 = np.kron(ketp, np.kron(ketp, np.kron(ketp, ket0)))
    
    ketpp = np.kron(ketp, ketp)
    ketpppp = np.kron(ketp, np.kron(ketp, np.kron(ketp, ketp)))
    
    def print_init_tests():
        global tic
        global num_tests
        print("Running tests ...")
        print("---------------------------------------------------")
        tic = time.time()
        num_tests = 0
        return
    
    def pre_test_task():
        global num_tests
        global q1
        q1 = None
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
    exp_st_vec = np.array([1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)])
    q1.init_state(init_type="CUSTOM", st_vec=exp_st_vec)
    assert((q1.st_vec == exp_st_vec).all()), "Test 4 failed."
    
    #Test 5	- State initialization - Initialize 2 qubit state to user defined value with real probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    exp_st_vec = np.array([1/np.sqrt(13), np.sqrt(2)/np.sqrt(13), -1*np.sqrt(3)/np.sqrt(13), np.sqrt(7)/np.sqrt(13)])
    q1.init_state(init_type="CUSTOM", st_vec=exp_st_vec)
    assert((q1.st_vec == exp_st_vec).all()), "Test 5 failed."
    
    #Test 6	- State initialization - Initialize 4 qubit state to user defined value with real probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    exp_st_vec = np.array([1/np.sqrt(25), -1*1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25),
                     1/np.sqrt(25), 1/np.sqrt(25), -1*1/np.sqrt(25), 1/np.sqrt(25),
                     1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25), -1*1/np.sqrt(25),
                     -1*1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25), np.sqrt(10)/np.sqrt(25)])
    q1.init_state(init_type="CUSTOM", st_vec=exp_st_vec)
    assert((q1.st_vec == exp_st_vec).all()), "Test 6 failed."

    #Test 7	- State initialization - Initialize 1 qubit state to user defined value with complex probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1, tolerance=10**(-8))
    exp_st_vec = np.array([1/np.sqrt(3), (np.sqrt(2)/np.sqrt(3))*1j])
    q1.init_state(init_type="CUSTOM", st_vec=exp_st_vec)
    assert((q1.st_vec == exp_st_vec).all()), "Test 7 failed."
    
    #Test 8	- State initialization - Initialize 2 qubit state to user defined value with complex probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2, tolerance=10**(-8))
    exp_st_vec = np.array([0.12212234-0.59549103j, -0.30416912+0.29570485j,
                     0.46135097-0.45942149j,  0.00583572-0.16300147j])
    q1.init_state(init_type="CUSTOM", st_vec=exp_st_vec)
    assert((q1.st_vec == exp_st_vec).all()), "Test 8 failed."
    
    #Test 9	- State initialization - Initialize 4 qubit state to user defined value with complex probablity amplitudes
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4, tolerance=10**(-8))
    exp_st_vec = np.array([0.21584501+0.2494106j ,  0.05495676+0.03468601j,
                     -0.05731732+0.15385203j,  0.1354276 -0.13732148j,
                     0.24636623-0.26832918j,  0.19739793+0.19480392j,
                     -0.04900181-0.02107175j,  0.11115504-0.21241653j,
                     0.24895018+0.26924763j,  0.11261091-0.15254312j,
                     0.04105645-0.22667174j, -0.21916044-0.28315859j,
                     -0.01293583+0.09801503j, -0.15619437-0.21974565j,
                     -0.25840702+0.15058432j, -0.06511822-0.15826991j])
    q1.init_state(init_type="CUSTOM", st_vec=exp_st_vec)
    assert((q1.st_vec == exp_st_vec).all()), "Test 9 failed."
    
    #Test 10 - State initialization - Initialize 1 qubit state to ground state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    exp_st_vec = np.zeros(2) + np.zeros(2)*1j
    exp_st_vec[0] = 1
    q1.init_state(init_type="GROUND")
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 10 failed."
    
    #Test 11 - State initialization - Initialize 2 qubit state to ground state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    exp_st_vec = np.zeros(4) + np.zeros(4)*1j
    exp_st_vec[0] = 1
    q1.init_state(init_type="GROUND")
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 11 failed."
    
    #Test 12 - State initialization	- Initialize 4 qubit state to ground state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    exp_st_vec = np.zeros(16) + np.zeros(16)*1j
    exp_st_vec[0] = 1
    q1.init_state(init_type="GROUND")
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 12 failed."
    
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
    exp_st_vec = (1/np.sqrt(2))*np.array([1, 1])
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 16 failed."
        
    q1.measure_ckt(num_shots=10000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0" : 5000, "1" : 5000}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 16 failed."
        
    #Next, to the state |1>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    psi_init = np.array([0, 1])
    exp_st_vec = (1/np.sqrt(2))*np.array([1, -1])
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 16 failed."
        
    q1.measure_ckt(num_shots=10000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0" : 5000, "1" : 5000}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 16 failed."       
    
    #Test 17 - Quantum gates and measurement - Apply H to qubit 0 of 2 qubit state
    pre_test_task()
    #To the state |00>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket0p
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 17 failed."
        
    q1.measure_ckt(num_shots=10000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 5000, "01" : 5000, "10" : 0, "11" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 17 failed."
        
    #Test 18 - Quantum gates and measurement - Apply H to qubit 1 of 2 qubit state
    pre_test_task()
    #To the state |01>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec = ket01)
    exp_st_vec = ketp1
    program = [{"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 18 failed."
        
    q1.measure_ckt(num_shots=10000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 0, "01" : 5000, "10" : 0, "11" : 5000}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 18 failed."
    #Now to the state |10>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec = ket10)
    exp_st_vec = ketm0
    program = [{"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 18 failed."
        
    q1.measure_ckt(num_shots=10000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 5000, "01" : 0, "10" : 5000, "11" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 18 failed."
        
    #Test 19 - Quantum gates and measurement -	Apply H to qubits 0 and 1 of 2 qubit state
    #To the state |00>
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketpp
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 19 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 5000, "01" : 5000, "10" : 5000, "11" : 5000}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 19 failed."
        
    #Test 20 - Quantum gates and measurement - Apply H to qubit 0 of 4 qubit state
    #To the state |0000>
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket000p
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 20 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 10000, 
                "0001" : 10000,
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
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 20 failed."
        
    #Test 21 - Quantum gates and measurement - Apply H to qubit 1 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket00p0
    program = [{"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 21 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 10000, 
                "0001" : 0,
                "0010" : 10000,
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
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 21 failed."
        
    #Test 22 - Quantum gates and measurement - Apply H to qubit 2 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket0p00
    program = [{"gate_type" : "H", "target" : 2}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 22 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 10000, 
                "0001" : 0,
                "0010" : 0,
                "0011" : 0,
                "0100" : 10000, 
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
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 22 failed."
        
    #Test 23 - Quantum gates and measurement - Apply H to qubit 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketp000
    program = [{"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 23 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 10000, 
                "0001" : 0,
                "0010" : 0,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 10000, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 23 failed."
        
    #Test 24 - Quantum gates and measurement - Apply H to qubits 0 and 1 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket00pp
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 24 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 5000,
                "0010" : 5000,
                "0011" : 5000,
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
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 24 failed."
        
    #Test 25 - Quantum gates and measurement - Apply H to qubits 1 and 2 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket0pp0
    program = [{"gate_type" : "H", "target" : 1}, {"gate_type" : "H", "target" : 2}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 25 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 0,
                "0010" : 5000,
                "0011" : 0,
                "0100" : 5000, 
                "0101" : 0,
                "0110" : 5000,
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
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 25 failed."
        
    #Test 26 - Quantum gates and measurement - Apply H to qubits 2 and 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketpp00
    program = [{"gate_type" : "H", "target" : 2}, {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 26 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 0,
                "0010" : 0,
                "0011" : 0,
                "0100" : 5000, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 5000, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 5000, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 26 failed."
        
    #Test 27 - Quantum gates and measurement - Apply H to qubits 0 and 2 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket0p0p
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 2}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 27 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 5000,
                "0010" : 0,
                "0011" : 0,
                "0100" : 5000, 
                "0101" : 5000,
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
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 27 failed."
        
    #Test 28 - Quantum gates and measurement - Apply H to qubits 1 and 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketp0p0
    program = [{"gate_type" : "H", "target" : 1}, {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 28 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 0,
                "0010" : 5000,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 5000, 
                "1001" : 0,
                "1010" : 5000,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 28 failed."
        
    #Test 29 - Quantum gates and measurement - Apply H to qubits 0 and 3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketp00p
    program = [{"gate_type" : "H", "target" : 0}, {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 29 failed."
        
    q1.measure_ckt(num_shots=20000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 5000,
                "0010" : 0,
                "0011" : 0,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 5000, 
                "1001" : 5000,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 29 failed."
        
    #Test 30 - Quantum gates and measurement - Apply H to qubits 0,1,2 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket0ppp
    program = [{"gate_type" : "H", "target" : 0}, 
               {"gate_type" : "H", "target" : 1}, 
               {"gate_type" : "H", "target" : 2}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 30 failed."
        
    q1.measure_ckt(num_shots=40000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 5000,
                "0010" : 5000,
                "0011" : 5000,
                "0100" : 5000, 
                "0101" : 5000,
                "0110" : 5000,
                "0111" : 5000,
                "1000" : 0, 
                "1001" : 0,
                "1010" : 0,
                "1011" : 0,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 30 failed."
        
    #Test 31 - Quantum gates and measurement - Apply H to qubits 0,2,3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketpp0p
    program = [{"gate_type" : "H", "target" : 0}, 
               {"gate_type" : "H", "target" : 2}, 
               {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 31 failed."
        
    q1.measure_ckt(num_shots=40000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 5000,
                "0010" : 0,
                "0011" : 0,
                "0100" : 5000, 
                "0101" : 5000,
                "0110" : 0,
                "0111" : 0,
                "1000" : 5000, 
                "1001" : 5000,
                "1010" : 0,
                "1011" : 0,
                "1100" : 5000, 
                "1101" : 5000,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 31 failed."
        
    #Test 32 - Quantum gates and measurement - Apply H to qubits 0,1,3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketp0pp
    program = [{"gate_type" : "H", "target" : 0}, 
               {"gate_type" : "H", "target" : 1}, 
               {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 32 failed."
        
    q1.measure_ckt(num_shots=40000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 5000,
                "0010" : 5000,
                "0011" : 5000,
                "0100" : 0, 
                "0101" : 0,
                "0110" : 0,
                "0111" : 0,
                "1000" : 5000, 
                "1001" : 5000,
                "1010" : 5000,
                "1011" : 5000,
                "1100" : 0, 
                "1101" : 0,
                "1110" : 0,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 32 failed."
        
    #Test 33 - Quantum gates and measurement - Apply H to qubits 1,2,3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketppp0
    program = [{"gate_type" : "H", "target" : 1}, 
               {"gate_type" : "H", "target" : 2}, 
               {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 33 failed."
        
    q1.measure_ckt(num_shots=40000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 5000, 
                "0001" : 0,
                "0010" : 5000,
                "0011" : 0,
                "0100" : 5000, 
                "0101" : 0,
                "0110" : 5000,
                "0111" : 0,
                "1000" : 5000, 
                "1001" : 0,
                "1010" : 5000,
                "1011" : 0,
                "1100" : 5000, 
                "1101" : 0,
                "1110" : 5000,
                "1111" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 33 failed."
        
    #Test 34 - Quantum gates and measurement - Apply H to qubits 0,1,2,3 of 4 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ketpppp
    program = [{"gate_type" : "H", "target" : 0},
               {"gate_type" : "H", "target" : 1}, 
               {"gate_type" : "H", "target" : 2}, 
               {"gate_type" : "H", "target" : 3}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 34 failed."
        
    q1.measure_ckt(num_shots=40000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 2500, 
                "0001" : 2500,
                "0010" : 2500,
                "0011" : 2500,
                "0100" : 2500, 
                "0101" : 2500,
                "0110" : 2500,
                "0111" : 2500,
                "1000" : 2500, 
                "1001" : 2500,
                "1010" : 2500,
                "1011" : 2500,
                "1100" : 2500, 
                "1101" : 2500,
                "1110" : 2500,
                "1111" : 2500}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 34 failed."
        
    #Test 35 - Quantum gates and measurement - Apply I gate to qubit 1 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket00
    program = [{"gate_type" : "I", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 35 failed."
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 2000,
                "01" : 0,
                "10" : 0,
                "11" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 35 failed."
        
    #Test 36 - Quantum gates and measurement - Apply X gate to qubit 1 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    exp_st_vec = ket10
    program = [{"gate_type" : "X", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 36 failed."
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 0,
                "01" : 0,
                "10" : 2000,
                "11" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 36 failed."
        
    #Test 37 - Quantum gates and measurement - Apply Y gate to qubit 1 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    exp_st_vec = np.array([0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j])
    program = [{"gate_type" : "Y", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 37 failed."
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 0,
                "01" : 0,
                "10" : 2000,
                "11" : 0}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 37 failed."
        
    #Test 38 - Quantum gates and measurement - Apply Z gate to qubit 1 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    psi_init = (1/np.sqrt(2))*(ket00 + ket11)
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    exp_st_vec = (1/np.sqrt(2))*(ket00 - ket11)
    program = [{"gate_type" : "Z", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 38 failed."
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 1000,
                "01" : 0,
                "10" : 0,
                "11" : 1000}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 38 failed."
        
    #Test 39 - Quantum gates and measurement - Apply S gate to qubit 1 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    psi_init = (1/np.sqrt(2))*(ket00 + ket11)
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    exp_st_vec = (1/np.sqrt(2))*(ket00 + (ket11*1j))
    program = [{"gate_type" : "S", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 39 failed."
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 1000,
                "01" : 0,
                "10" : 0,
                "11" : 1000}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 39 failed."
        
    #Test 40 - Quantum gates and measurement - Apply T gate to qubit 1 of 2 qubit state  
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    psi_init = (1/np.sqrt(2))*(ket00 + ket11)
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    exp_st_vec = np.array([0.70710678+0.j , 0.+0.j , 0.+0.j , 0.5-0.5j])
    program = [{"gate_type" : "T", "target" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-8)), "Test 40 failed."
    q1.measure_ckt(num_shots=2000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 1000,
                "01" : 0,
                "10" : 0,
                "11" : 1000}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 40 failed."
        
    #Test 41 - Quantum state with complex probability amplitudes - Prepare and measure arbitrary 1 qubit state with complex coefficients
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1, tolerance=10**(-6))
    psi_init = np.array([ 0.58828644-0.36892633j, -0.51168687-0.50595352j])
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    #This is the expected output
    exp_st_vec = np.array([0.05416408-0.61863348j, 0.77779859+0.09689285j])
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 41 failed."
    q1.measure_ckt(num_shots=1000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0" : 386,
                "1" : 614}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 41 failed."
        
    #Test 42 - Quantum state with complex probability amplitudes - Prepare and measure arbitrary 2 qubit state with complex coefficients
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2, tolerance=10**(-6))
    psi_init = np.array([-0.13562542-0.38181362j,  0.33140125+0.59110471j,
        0.41353633+0.2101598j , -0.3641401 -0.16975008j])
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    #This is the expected output
    exp_st_vec = np.array([ 0.13843442+0.14799115j, -0.33023773-0.68795715j,
        0.03492841+0.02857399j,  0.54990028+0.26863685j])
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 42 failed."
    q1.measure_ckt(num_shots=100000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 4100,
                "01" : 58200,
                "10" : 200,
                "11" : 37500}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= 2*meas_tol*exp_prob[key]), "Test 42 failed."
        
    #Test 43 - Quantum state with complex probability amplitudes - Prepare and measure arbitrary 4 qubit state with complex coefficients
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4, tolerance=10**(-6))
    psi_init = np.array([-0.30918978+0.21020209j,  0.17072729+0.3023967j ,
        0.19661994-0.23847216j,  0.14237429+0.11949326j,
        0.05580431-0.07050416j, -0.261705  -0.25702139j,
        0.05264001-0.21342211j, -0.26517493+0.18882867j,
       -0.109402  +0.10429205j,  0.13688214+0.20035965j,
        0.0427491 -0.02871469j, -0.21629255-0.01129443j,
        0.08293346+0.10800028j, -0.00881172+0.23182094j,
       -0.13924875-0.02953286j,  0.29047251+0.06583015j])
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    #This is the expected output
    exp_st_vec = np.array([-0.09790776+0.36246208j, -0.33935261-0.06519143j,
        0.23970512-0.08413079j,  0.03835746-0.25311978j,
       -0.14559378-0.23159554j,  0.22451299+0.1318876j ,
       -0.15028489-0.01739019j,  0.2247291 -0.28443425j,
        0.0194314 +0.21542128j, -0.17414919-0.06793005j,
       -0.12271375-0.02829072j,  0.18317011-0.01231798j,
        0.05241199+0.24028988j,  0.06487364-0.08755443j,
        0.10693134+0.02566606j, -0.30385882-0.06743183j])
    program = [{"gate_type" : "H", "target" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 43 failed."
    q1.measure_ckt(num_shots=100000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0000" : 14096, 
                "0001" : 11941,  
                "0010" : 6454,  
                "0011" : 6554,
                "0100" : 7483,
                "0101" : 6780,
                "0110" : 2289,
                "0111" : 13141,
                "1000" : 4678,
                "1001" : 3494,
                "1010" : 1586,
                "1011" : 3370,
                "1100" : 6049,
                "1101" : 1187,
                "1110" : 1209,
                "1111" : 9688}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 43 failed."
        
    #Test 44 - Quantum gates (parametric) and measurement - Apply Parametric gate to 1 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1, tolerance=10**(-6))
    #U1 gate
    my_u1 = [["1", "0"],["0", "exp(alpha * 1.0j)"]]
    my_u1_params = {"alpha" : 1.0471975511965976} #alpha = pi/3
    program = [{"gate_type" : "PARAMETRIC", "target" : 0, "control" : None, 
                "parametric_gate" : my_u1, "parameters" : my_u1_params}]
    psi_init = ketp
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    #This is the expected output state
    exp_st_vec = np.array([0.70710678+0.j, 0.35355339+0.61237244j])
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 44 failed."
    q1.measure_ckt(num_shots=1000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"0" : 500,
                "1" : 500}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 44 failed."
           
    #Test 45 - Quantum gates (parametric) and measurement - Apply parametric gate to qubit 0 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2, tolerance=10**(-6))
    #U3 gate - parameters theta, alpha, phi
    my_u3 = [["cos(0.5 * theta)", "-1 * exp(alpha * 1j) * sin(0.5 * theta)"],["exp(phi * 1j) * sin(0.5 * theta)", "exp((phi + alpha) * 1.0j) * cos(0.5 * theta)"]]
    my_u3_params = {"theta" : 1.0471975511965976,
                    "phi"   : 1.0471975511965976,
                    "alpha" : 1.0471975511965976} #all pi/3
    psi_init = ketpp
    program = [{"gate_type" : "PARAMETRIC", "target" : 0, "control" : None, 
                "parametric_gate" : my_u3, "parameters" : my_u3_params}]
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    #This is the expected output state
    exp_st_vec = np.array([0.3080127 -0.21650635j, -0.09150635+0.59150635j,
                     0.3080127 -0.21650635j, -0.09150635+0.59150635j])
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 45 failed."
    q1.measure_ckt(num_shots=10000)
    #If measured frequency are off by more than 10% compared to expected 
    #frequency, issue an error
    exp_prob = {"00" : 1420,
                "01" : 3580,
                "10" : 1420,
                "11" : 3580}
    for key in exp_prob:
        assert(np.abs(q1.results[key] - exp_prob[key]) <= meas_tol*exp_prob[key]), "Test 45 failed."
    
    #Test 46 - Quantum gates (controlled) - Apply Controlled X gate on control=1, target=0 of 2 qubit state
    pre_test_task()
    #init with |00>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec=np.array([1,0,0,0]))
    #expected state
    exp_st_vec = np.array([1,0,0,0])
    program = [{"gate_type" : "X", "target" : 0, "control" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 46 failed."
        
    #init with |01>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,1,0,0]))
    #expected state
    exp_st_vec = np.array([0,1,0,0])
    program = [{"gate_type" : "X", "target" : 0, "control" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 46 failed."
        
    #init with |10>
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,1,0]))
    #expected state
    exp_st_vec = np.array([0,0,0,1])
    program = [{"gate_type" : "X", "target" : 0, "control" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 46 failed."

    #init with |11>        
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,1]))
    #expected state
    exp_st_vec = np.array([0,0,1,0])
    program = [{"gate_type" : "X", "target" : 0, "control" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 46 failed."
    
    #Test 47 - Quantum gates (controlled) - Apply Controlled X gate on control=0, target=1 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    #init with |11>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,1]))
    #expected state
    exp_st_vec = np.array([0,1,0,0])
    program = [{"gate_type" : "X", "target" : 1, "control" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 47 failed."
        
    #Test 48 - Quantum gates (controlled) - Apply Controlled Z gate on control=1, target=0 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    #init with |11>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,1]))
    #expected state
    exp_st_vec = np.array([0,0,0,-1])
    program = [{"gate_type" : "Z", "target" : 0, "control" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 48 failed."
        
    #Test 49 - Quantum gates (controlled) - Apply Controlled X gate on control=1, target=0 of 3 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=3)
    #init with |111>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,0,0,0,0,1]))
    #expected state
    exp_st_vec = np.array([0,0,0,0,0,0,1,0])
    program = [{"gate_type" : "X", "target" : 0, "control" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 49 failed."
        
    #Test 50 - Quantum gates (controlled) - Apply Controlled X gate on control=0, target=1 of 3 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=3)
    #init with |011>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,1,0,0,0,0]))
    #expected state
    exp_st_vec = np.array([0,1,0,0,0,0,0,0])
    program = [{"gate_type" : "X", "target" : 1, "control" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 50 failed."
        
    #Test 51 - Quantum gates (controlled) - Apply Controlled X gate on control=2, target=1 of 3 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=3)
    #init with |110>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,0,0,0,1,0]))
    #expected state
    exp_st_vec = np.array([0,0,0,0,1,0,0,0])
    program = [{"gate_type" : "X", "target" : 1, "control" : 2}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 51 failed."
    
    #Test 52 - Quantum gates (controlled) - Apply Controlled X gate on control=1, target=2 of 3 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=3)
    #init with |110>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,0,0,0,1,0]))
    #expected state
    exp_st_vec = np.array([0,0,1,0,0,0,0,0])
    program = [{"gate_type" : "X", "target" : 2, "control" : 1}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 52 failed."
    
    #Test 53 - Quantum gates (controlled) - Apply Controlled X gate on control=2, target=0 of 3 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=3)
    #init with |110>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,0,0,0,0,0,1,0]))
    #expected state
    exp_st_vec = np.array([0,0,0,0,0,0,0,1])
    program = [{"gate_type" : "X", "target" : 0, "control" : 2}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 53 failed."
        
    #Test 54 - Quantum gates (controlled) - Apply Controlled X gate on control=0, target=2 of 3 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=3)
    #init with |001>
    q1.init_state(init_type="CUSTOM", st_vec=np.array([0,1,0,0,0,0,0,0]))
    #expected state
    exp_st_vec = np.array([0,0,0,0,0,1,0,0])
    program = [{"gate_type" : "X", "target" : 2, "control" : 0}]
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 54 failed."
    
    #Test 55 - Quantum gates (parametric) - Apply Controlled parametric gate on control=1, target=0 of 2 qubit state
    pre_test_task()
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2, tolerance=10**(-6))
    #U1 gate
    my_u1 = [["1", "0"],["0", "exp(alpha * 1.0j)"]]
    my_u1_params = {"alpha" : 1.0471975511965976} #alpha = pi/3
    program = [{"gate_type" : "PARAMETRIC", "target" : 0, "control" : 1, 
                "parametric_gate" : my_u1, "parameters" : my_u1_params}]
    psi_init = np.kron(np.array([0,1]),ketp)
    q1.init_state(init_type="CUSTOM", st_vec=psi_init)
    #This is the expected output state
    exp_st_vec = np.kron(np.array([0+(0*1j),1+(0*1j)]),np.array([0.70710678+(0*1j), 0.35355339+(0.61237244*1j)]))
    q1.execute_ckt(program)
    for i in range(len(exp_st_vec)):
        assert(np.linalg.norm(exp_st_vec[i] - q1.st_vec[i]) < 10**(-6)), "Test 55 failed."

    print_report_tests()
    return
