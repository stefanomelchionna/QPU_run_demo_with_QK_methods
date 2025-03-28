# QPU_run_demo_with_QK_methods

In this tutorial, we would like to show an example of how to run code on a quantum computer. Coding a quantum computer can be done via an API using the Python language.

Depending on the task, coding a quantum computer can be done at a very abstract level, e.g., by simply calling some already implemented well-known algorithms. Alternatively, one can code very close to the circuit level, specifying operations at the qubit and gate level.

In the notebook, we see an example of rather abstract coding. We would like to apply some popular Quantum Machine Learning (QML) methods known as Quantum Kernel Methods to an anomaly detection problem. Quantum Kernel Methods are analogous to classic Kernel Methods, with the only difference being the kernel matrix is computed via a quantum circuit. For more information on classic Kernel Methods see https://en.wikipedia.org/wiki/Kernel_method.

The notebook is structured as follows:

In the first part, we explain the anomaly detection problem we want to solve.
Second, we fit a one-class support vector machine (OCSVM) with a classic (Gaussian) kernel to our data and evaluate the performance.
Third, we fit an OCSVM with a quantum kernel, which is computed on a local simulator. We then evaluate the performance and compare it to the classical case.
Finally, we connect to the IBM cloud and fit an OCSVM with a quantum kernel computed on a real quantum computer.
We will see that to work with quantum computers, all we have to do is to simply add a few lines of code.

If you don’t know anything about machine learning or coding, don’t worry. All you have to understand from this notebook is that, in some circumstances, coding on a quantum computer requires very little additional effort and doesn’t necessarily require a deep understanding of quantum computing.

The .py file contains some auxiliary funtions used in the notebook to load preprocess and diyplay data. 
