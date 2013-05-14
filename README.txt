
PyNFG is a Python package for modeling and solving Network Form Games. It is distributed under the GNU Affero GPL. http://www.gnu.org/licenses/agpl.html

1. Welcome
--------------------
PyNFG is designed to make it easy for researchers to model strategic environments using the Network Form Game (NFG) formalism developed by David Wolpert with contributions from Ritchie Lee, James Bono and others. The main idea of the NFG framework is to translate a strategic environment into the language of probabilistic graphical models. The result is a more intuitive, powerful, and user-friendly framework than the extensive form.

For an introduction to the semi-NFG framework and Level-K D-Relaxed Strategies:

- Lee, R. and Wolpert, D.H., “Game-Theoretic Modeling of Human Behavior in Mid-Air Collisions”,  Decision-Making with Imperfect Decision Makers, T. Guy, M. Karny and D.H.Wolpert (Ed.’s), Springer (2011).

For an introduction to iterated semi-NFG framework and Level-K Reinforcement Learning:

- Ritchie Lee, David H. Wolpert, James Bono, Scott Backhaus, Russell Bent, Brendan Tracey. "Counter-Factual Reinforcement Learning: How to Model Decision-Makers That Anticipate The Future."  http://arxiv.org/abs/1207.0852

- Scott Backhaus, Russell Bent, James Bono, Ritchie Lee, Brendan Tracey, David Wolpert, Dongping Xie, Yildiray Yildiz "Cyber-Physical Security: A Game Theory Model of Humans Interacting over Control Systems." http://arxiv.org/abs/1304.3996

For an introduction to Predictive Game Theory:

- David Wolpert and James Bono "Distribution-Valued Solution Concepts" http://ssrn.com/abstract=1622463

- James Bono and David Wolpert "Decision-Theoretic Prediction and Policy Design of GDP Slot Auctions" http://ssrn.com/abstract=1815222

2. Installation
--------------------
PyNFG requires the following packages: Numpy, Scipy, Matplotlib, Networkx, and PyGraphviz. Pygraphviz and Networkx are used only for visualizing the Directed Acyclic Graphs (DAGs) that represent semi-NFGs. 

To install from source: Download the source from https://pypi.python.org/pypi/PyNFG/0.1.0. Unzip. Then from the directory with the unzipped files, do "python setup.py install".


3. Questions and Comments
----------------------------------
The documentation is hosted at http://pythonhosted.org/PyNFG/.

Please contact James Bono for questions about using PyNFG in your research, reporting bug fixes, offering suggestions, etc.


4. Contributors
----------------------
PyNFG is authored by James Bono with contributions by Dongping Xie. The project has received valuable feedback from Justin Grana, David Wolpert, Adrian Agogino, Juan Alonso, Brendan Tracey, Alice Fan, Dominic McConnachie, Kee Palopo, Huu Huynh, and others.
