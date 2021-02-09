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
#---------------------------------------------------

def launch_tests():
    
    num_tests=0
    print("Running tests ...")
    #Test 1	- State dimension - Quantum state dimension for 1 qubit state
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    q1.init_state(init_type="GROUND")
    assert(q1.st_dim == 2**1), "Test 1 failed."
    num_tests += 1
    
    #Test 2	- State dimension - Quantum state dimension for 2 qubit state
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    q1.init_state(init_type="GROUND")
    assert(q1.st_dim == 2**2), "Test 2 failed."
    num_tests += 1
    
    #Test 3	- State dimension - Quantum state dimension for 4 qubit state
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    q1.init_state(init_type="GROUND")
    assert(q1.st_dim == 2**4), "Test 3 failed."
    num_tests += 1
    
    #Test 4	- State initialization - Initialize 1 qubit state to user defined value with real probablity amplitudes
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1)
    psi1 = np.array([1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 4 failed."
    num_tests += 1
    
    #Test 5	- State initialization - Initialize 2 qubit state to user defined value with real probablity amplitudes
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2)
    psi1 = np.array([1/np.sqrt(13), np.sqrt(2)/np.sqrt(13), -1*np.sqrt(3)/np.sqrt(13), np.sqrt(7)/np.sqrt(13)])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 5 failed."
    num_tests += 1
    
    #Test 6	- State initialization - Initialize 4 qubit state to user defined value with real probablity amplitudes
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=4)
    psi1 = np.array([1/np.sqrt(25), -1*1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25),
                     1/np.sqrt(25), 1/np.sqrt(25), -1*1/np.sqrt(25), 1/np.sqrt(25),
                     1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25), -1*1/np.sqrt(25),
                     -1*1/np.sqrt(25), 1/np.sqrt(25), 1/np.sqrt(25), np.sqrt(10)/np.sqrt(25)])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 6 failed."
    num_tests += 1

    #Test 7	- State initialization - Initialize 1 qubit state to user defined value with complex probablity amplitudes
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=1, tolerance=10**(-8))
    psi1 = np.array([1/np.sqrt(3), (np.sqrt(2)/np.sqrt(3))*1j])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 7 failed."
    num_tests += 1
    
    #Test 8	- State initialization - Initialize 2 qubit state to user defined value with complex probablity amplitudes
    q1 = siqcandar.siqc_ckt(name="q1", num_qubits=2, tolerance=10**(-8))
    psi1 = np.array([0.12212234-0.59549103j, -0.30416912+0.29570485j,
                     0.46135097-0.45942149j,  0.00583572-0.16300147j])
    q1.init_state(init_type="CUSTOM", st_vec=psi1)
    assert((q1.st_vec == psi1).all()), "Test 8 failed."
    num_tests += 1
    
    #Test 9	- State initialization - Initialize 4 qubit state to user defined value with complex probablity amplitudes
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
    num_tests += 1
    
    print("Tests passed. Number of tests run = ", num_tests, ". If you see this line without encountering any errors, it means the tests have passed.")
    return
