# -*- coding: utf-8 -*-
#
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    #In DFS, we will use a stack to store the posible paths
    sol=util.Stack()
    #Stores the starting point, the path and the cost
    sol.push((problem.getStartState(),[],0))
    #Nodes we have visited
    expanded=[]
    #There's the possibility that there's not a path from the starting point to the end, so we
    #check if our stack is empty
    while not sol.isEmpty():
        curPos, path, cost=sol.pop()
        #If the current position is a goal state, we return the path
        if problem.isGoalState(curPos):
            return path
        #If we haven't visited this node yet, we can still continue following its path
        if not curPos in expanded:
            expanded.append(curPos)
            #Adds a new component for each successor of our current position
            for s, d, c in problem.getSuccessors(curPos):
                sol.push((s,path+[d],c+cost))
            
    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    #The same as in DFS, but with a queue
    sol=util.Queue()
    sol.push((problem.getStartState(),[],0))
    expanded=[]
    while not sol.isEmpty():
        curPos, path, cost=sol.pop()
        if problem.isGoalState(curPos):
            return path
        if not curPos in expanded:
            expanded.append(curPos)
            for s, d, c in problem.getSuccessors(curPos):
                sol.push((s,path+[d],c+cost))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # This problem will be almost identical to BFS, but now, we will use a priority queue
    # That will look at the distances between nodes, to find the less "expensive" path
    sol=util.PriorityQueue()
    sol.push((problem.getStartState(),[],0),0)
    expanded=[]
    while not sol.isEmpty():
        curPos, path, cost=sol.pop()
        if problem.isGoalState(curPos):
            return path
        if not curPos in expanded:
            expanded.append(curPos)
            for s, d, c in problem.getSuccessors(curPos):
                sol.push((s,path+[d],c+cost),(cost+c+heuristic(s,problem)))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
