# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:39:38 2021

Motivation: An example of the circuit drawing function of SIQCANDAR simulator.

@author: Dhruv Bhatnagar
"""

#The simulator uses python turtle to implement a circuit drawer for the 
#initialization, supported gates and measurement. Try to run this program. If
#it runs succesfully, a figure similar to demo2_fig.png should open up.

import siqcandar as siqc

qc = siqc.siqc_ckt(name="qc", num_qubits=4)
qc.init_state(init_type="GROUND")
my_u3 = [["cos(0.5 * theta)",
          "-1 * exp(alpha * 1j) * sin(0.5 * theta)"],
         ["exp(phi * 1j) * sin(0.5 * theta)", 
          "exp((phi + alpha) * 1.0j) * cos(0.5 * theta)"]]
my_u3_params = {"theta" : 1.0471975511965976,
                "phi"   : 1.0471975511965976,
                "alpha" : 1.0471975511965976} #all pi/3
program = [{"gate_type" : "H", "target" : 0},
           {"gate_type" : "Y", "target" : 2},
           {"gate_type" : "PARAMETRIC", "target" : 0, "control" : 2, 
            "parametric_gate" : my_u3, "parameters" : my_u3_params},
           {"gate_type" : "Z", "target" : 3, "control" : 0}]
qc.execute_ckt(program)
qc.measure_ckt(num_shots=1000)
qc.draw_ckt()
