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
        # Useful information you can extract from a GameState (pacman.py)
        if action == 'Stop':return -float("inf")
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghosts=len(newGhostStates)
        for i in range(ghosts):
            if newGhostStates[i].getPosition()==newPos and newScaredTimes[i] == 0:
                return -float("inf")
        dist=[]
        for i in currentGameState.getFood().asList():
            dist.append(manhattanDistance(i,newPos))
        return 1000 - min(dist)

      
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

    def minimax(self,gameState, depth, index, agents):
        if depth==0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent=(index+1)%agents
        if not index:
            alpha = -float('inf')
            for i in gameState.getLegalActions(index):
                alpha = max(alpha, self.minimax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents))
            return alpha
        else:
            beta = float('inf')
            for i in gameState.getLegalActions(index):
                beta = min(beta, self.minimax(gameState.generateSuccessor(index, i), depth - 1, newAgent,agents))
            return beta


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

    def minimaxPrun(self,gameState, depth, index, agents,alpha,beta):
        if depth==0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent=(index+1)%agents
        if not index:
            v = -float('inf')
            for i in gameState.getLegalActions(index):
                v = max(v, self.minimaxPrun(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents, alpha, beta))
                if v>beta:
                    return v
                alpha=max(v,alpha)
            return v
        else:
            v = float('inf')
            for i in gameState.getLegalActions(index):
                v = min(v, self.minimaxPrun(gameState.generateSuccessor(index, i), depth - 1, newAgent,agents, alpha, beta))
                if v<alpha:
                    return v
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

    def expectiminimax(self, gameState, depth, index, agents):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        newAgent = (index + 1) % agents
        if not index:
            alpha = -float('inf')
            for i in gameState.getLegalActions(index):
                alpha = max(alpha, self.expectiminimax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents))
            return alpha
        else:
            beta = 0
            actions=0
            for i in gameState.getLegalActions(index):
                actions+=1
                beta = beta+self.expectiminimax(gameState.generateSuccessor(index, i), depth - 1, newAgent, agents)
            return float(beta/actions)

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
        # print ('dist', dist)
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
            ghostScore -= 1.0 / (3.0 - dist);
        elif state.scaredTimer < dist:
            ghostScore += 1.0 / (dist)

    score += 1.0 / (1 + len(newFood.asList())) + 1.0 / (1 + foodDist) + ghostScore + currentGameState.getScore()

    return score;


# Abbreviation
better = betterEvaluationFunction

