### qosf_appl_feb_2021_work
Author: Dhruv Bhatnagar.

This is my submission to the screening task for QOSF mentorship program (application period - February 2021)

Task attempted - Task 3

The task description goes as follows:
"Learning by doing: the best way to understand the basics of quantum computation is to implement a quantum circuit simulator. This task is suitable both for people from computer sciences who want to learn about quantum computing, and for people from math/physics who want to exercise coding

It is expected that simulator can perform following tasks:
* Initialize state
* read program, and for each gate:
  * calculate matrix operator
  * apply operator (modify state)
* perform multi-shot measurement of all qubits using weighted random technique
"

The following resources were shared with this, and were immensely helpful as references:

www.github.com/quantastica/qosf-mentorship/blob/master/qosf-simulator-task.ipynb

www.github.com/quantastica/qosf-mentorship/blob/master/qosf-simulator-task-aditional-info.pdf

These resources are gratefully acknowledged.

The name I chose for the simulator is SIQCANDAR - Simulator of Quantum Computations for Amateurs, Novices, Developers And Researchers. The name is inspired by the Persian version of the name of Alexander the Great.

Read the following points to find out how I have implemented the requested features. The motivation for my choice of the architecture of the simulator is as follows:

* Modeling the quantum state - Complex numpy arrays are used for state vectors to be able to model the most general states.
* Assertions - Since there are multiple calculations involved, I have added 'assert' checks in the simulator to be able to detect incorrect dimensions, states which are not normalized etc.
* Little Endianness convention is followed, ie |q_3 q_2 q_1 q_0>
* Quantum gates - modeled as 2-D numpy arrays. 
  * At present, gates acting on single qubits (in a single step of the program) are supported. 
  * Controlled gates are also supported - with the control being from a single qubit.
  * Parametric gates can be input as a list of matrix elements in string form. The parameter names can be quite general, and will be parsed by the simulator to calculate the numerical gate matrix.
  * Parametric gates controlled by a single qubit and acting on a single target are also supported.
  * The available gates in the simulator are single qubit H, X, Y, Z, I(Identity), S, T and parametric gates.
  * The simulator generates the gate matrices via appropriate tensor product expressions of single qubit gate matrices
* Measurement: To simulate random quantum measurements in multiple shots, it was required to implement a sampling method which could return random outcomes based on the probability distribution formed by using the quantum probability amplitudes (weighted random sampling). Since I wasn't aware that python has an implementation of this available, I coded my own :D. 
  * My logic behind my implementation weighted random sampling is as follows. Suppose we want to generate random outcomes according to the probability distribution 70% and 30% for 2 possible outcomes. The probability mass function is {0.7, 0.3}. The cumulative distribution function (cdf) is {0.7, 1.0}. Generate a uniform random number using python. Check if it is less than or equal to first entry of cdf 0.7 (this happens with probability 70% due to uniform distribution). If yes, return the random outcome as event 1. Else, check if it lies between 0.7 and 1.0 (this happens with probability 30%) and return the outcome as event 2. Thus, this generates random outcomes based on the provided probability distribution. A similar logic for multiple qubit-state measurement is implemented.
* Quantum state can be initialized to ground state, user-provided custom state and  simulator-generated random state. A method useful to obtain the vector corresponding to a ket labelled by a binary string is also implemented (for eg to get the state |00> by passing "00" as an argument)
* The program to be specified can be done via a list of dictionaries; see demos and tests.
* Measurement can be done in multiple shots
* User can pass parametric expressions for such gates - these expressions are not hard coded inside the simulator, so that the user can implement possibly any form of (unitary, single qubit) parametric gate. These expressions can be complex as well.
* A variational-like algorithm has also been demonstrated.
* All classes and functions have been documented


Capabilities implemented in addition to the mentioned tasks:

  * Method to return Bloch angles of  single qubit state
  * Controlled gates: single-qubit gates which are not only CNOT but other controlled gates are supported as well.
  * Statevector visualizer: a graphical way to visualize the probability amplitudes (via the height) and phases (via colour coding) of a quantum state vector
  * Measurements can be read as percentages or actual counts
  * Measurements can be visualized via a method to plot a bar diagram
  * The simulator can maintain a history of the states that the circuit has been in as a result of initialization or application of gates. 
  * The simulator can also keep track of the operations applied to the circuit.
  * A single qubit-controlled parameterized gate acting on a single qubit is also supported.
  * An elementary method to draw the circuit is also implemented using python turtle. It is intended to be as generic as possible for the implemented gate types. Currently it only works in .py files. See the file demo2.py for an example code and demo2_fig.png for a circuit generated by the simulator.


Testing:

* In order to be extremely confident of the implementation of the simulator, I decided to devote a significant amount of time to testing. Therefore, I added around 50+ self-checking tests, which are quite exhaustive in nature. These test cases are passing. If a change breaks some functionality, then these tests have the capability to return a failure, ie these should serve as a regression suite. (In rare cases, a test may fail due the random numbers generated by the simulator not conforming closely with the expected distribution. Such an outcome is possible in some cases, and should not be considrered a failure of functionality.)
* The test scenarios can be found in the test plan in the file test_plan.xlsx . They cover cases like multiple qubit states, gate operations, comparing obtained measurement outcomes with expected probability distributions, parametric gates etc.
* The tests are coded in test_scenarios.py and executed in tests_execute.ipynb
* These tests also helped me to find and fix multiple issues


Requirements:

numpy
matplotlib
sympy
turtle
time

File description:

~Code

siqcandar.py - source code for the simulator

~Tests

test_plan.xlsx - Tested scenarios in an excel sheet

test_scenarios.py - This is where the tests are coded along with helper functions

tests_execute.ipynb - Showing that the tests are passing

tests_execute.html - html version of the above notebook

~Examples and demonstration

demo1.ipynb - demonstrating usage and main capabilities of the simulator

demo1.html - html version of the above notebook

demo2.py - Example of code to generate the circuit drawing (ideally it should open in a popup. Try running 1-2 times.)

demo2_fig.png - figure of the code that the above example generated on my end.


If the jupter notebooks have problems in loading, please try the .html files. Otherwise, please try the nbviewer:

https://nbviewer.jupyter.org/github/dhruvbhq/qosf_appl_feb_2021_work/blob/main/demo1.ipynb

https://nbviewer.jupyter.org/github/dhruvbhq/qosf_appl_feb_2021_work/blob/main/tests_execute.ipynb
