# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        #To evaluate each action, we will consider three cases
        #The first will be if Pacman remains in the same place.
        #In that case, its evaluation will be -inf
        if action == 'Stop':return -float("inf")
        #If Pacman moves, we can start checking the rest of cases
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghosts=len(newGhostStates)
        #If the ghost can kill us moving to a given direction, we don't want it to go there
        #Therefore, its evaluation will be the same as if it was stopped
        for i in range(ghosts):
            if newGhostStates[i].getPosition()==newPos and newScaredTimes[i] == 0:
                return -float("inf")
            
        #Finally, we haver to get to the closest food, so we will calculate its distance to it and return the minimal
        return 1000 - min([manhattanDistance(i,newPos) for i in currentGameState.getFood().asList()])

      
######################
###End ReflexAgent####
######################

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        agents=gameState.getNumAgents()
        depth=agents*self.depth
        return max([(self.minimax(gameState.generateSuccessor(0,action),depth-1,1,agents),action) for action in gameState.getLegalActions(0)])[1]

    #We create a method to calculate minimax in a recursive way
    #It will receive current gamestate, the depth of the current graph, the current agent/index and the amount of agents
    def minimax(self,gameState, depth, index, agents):
        #If we have gotten to the end or we have lost/won the game, we will evaluate current gameState
        if depth==0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        #Pacman agent is 0, so if the current agent is Pacman, we have to look for a max
        if not index:
            return max([self.minimax(gameState.generateSuccessor(index, i), depth - 1, (index+1)%agents, agents) for i in gameState.getLegalActions(index)])
        #If its a ghost (index!=0), we will look for a minimum
        else:
            return min([self.minimax(gameState.generateSuccessor(index, i), depth - 1, (index+1)%agents,agents) for i in gameState.getLegalActions(index)])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agents=gameState.getNumAgents()
        alpha=-float('inf')
        beta=float('inf')
        m=-float('inf')
        depth=agents*self.depth
        for i in gameState.getLegalActions(0):
            v=self.minimaxPrun(gameState.generateSuccessor(0,i),depth-1,1,agents,alpha,beta)
            if v>m:
                action=i
                m=v
            if v>beta:
                return action
            alpha=max(v, alpha)
        return action

    #alpha and beta will store their current values in order to prune our tree
    def minimaxPrun(self,gameState, depth, index, agents,alpha,beta):
        #This function will work almost identical to minimax
        if depth==0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if not index:
            #v will store our current maximum
            #When we get to a value greater than our beta, we will prune the rest of nodes
            v = -float('inf')
            for i in gameState.getLegalActions(index):
                v = max(v, self.minimaxPrun(gameState.generateSuccessor(index, i), depth - 1, (index+1)%agents, agents, alpha, beta))
                if v>beta:
                    return v
                #If it's not, our current alpha will change
                alpha=max(v,alpha)
            return v
        else:
            #v will store our current minimum
            #When we get to a value smaller than our alpha, we will prune the rest of nodes
            v = float('inf')
            for i in gameState.getLegalActions(index):
                v = min(v, self.minimaxPrun(gameState.generateSuccessor(index, i), depth - 1, (index+1)%agents,agents, alpha, beta))
                if v<alpha:
                    return v
                #If it's not, our current beta will change
                beta=min(v,beta)
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        agents=gameState.getNumAgents()
        depth=agents*self.depth
        return max([(self.expectiminimax(gameState.generateSuccessor(0,action),depth-1,1,agents),action) for action in gameState.getLegalActions(0)])[1]

    #As before, it will work very similarly to minimax algorithm
    def expectiminimax(self, gameState, depth, index, agents):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent = (index + 1) % agents
        if not index:
            return max([self.expectiminimax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents) for i in gameState.getLegalActions(index)])
        else:
            #The difference between minimax and expectiminimax is that we will use a mean of all possible outputs
            beta = sum([self.expectiminimax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents) for i in gameState.getLegalActions(index)])
            return float(beta/len(gameState.getLegalActions(index)))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodDist = 0
    for food in newFood.asList():
        dist = manhattanDistance(food, newPos)
        foodDist += dist

    score = 0
    if len(newFood.asList()) == 0:
        score = 1000000000

    ghostScore = 0
    if newScaredTimes[0] > 0:
        ghostScore += 100.0
    for state in newGhostStates:
        dist = manhattanDistance(newPos, state.getPosition())
        if state.scaredTimer == 0 and dist < 3:
            ghostScore -= 1.0 / (3.0 - dist)
        elif state.scaredTimer < dist:
            ghostScore += 1.0 / (dist)

    score += 1.0 / (1 + len(newFood.asList())) + 1.0 / (1 + foodDist) + ghostScore + currentGameState.getScore()

    return score


# Abbreviation
better = betterEvaluationFunction

