# Ramificación y Acotamiento
This is an implementation of the Branch and bound algorithm for optimization problems like:
* Backpack problem
* Linear programming
* Integer, Binary and mixed Problems

This program displays a self explanatory UI that allows the user to input an optimization problem
to be solved. It will use the scipy linprog solver on the backend to solve individual steps of the
algorithm, but custom logic is used to determine how to branch and bound.
Finally the program will open an image of the solved branch and bound tree, which will also display
the optimal solution. This image is generated using the Pydot backend.

# Important

## Please keep in mind:
* This was a School project some time ago and was made with educational purposes in mind.
* Some of the code is really messy and most of it has Spanish all over it.
* It is not in my plans to maintain this project further as it has served its purpose.
* The project has been made public to serve as an example for people who might need ideas for  a similar project.
* You are free to fork this project and upgrade it.