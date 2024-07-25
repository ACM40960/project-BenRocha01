#Project done by
#Bernardo Rocha
#
#for the Projects in Maths Modelling module in UCD




#region Imports
import chess
import random
import math
import time
import numpy as np
import tensorflow as tf
#endregion





#region Global Variables
board = chess.Board()
library = 0
Bots = []

#endregion



 
 
#region Main functions 
    
def version():
    print("V 0.2.9")
    ###Changelog
    # V 0.1.0   -The whole evaluation system was revamped to be modular
    # V 0.2.0   -Code was reestructured. There are now classes and subclasses for most things.
    
def main():
    answer = "e"
    while answer != "q":
        print("\nWelcome to my chess player.\nChoose an option:")
        print("""
              --- Play against the computer (choose p)(in progress)
              --- Choosing the computer algorithm(choose a)(in progress)
              --- Make the computer play against itself (choose c)(in progress)
              --- list of branching methods (choose b)(in progress)
              --- list of evaluation functions (choose e)(in progress)
              --- tornament of all combinations (choose t)(in progress)
              --- quit (choose q)""")
        answer = input()
        
        if answer == "p":
            pass
        
        elif answer == "a":
            pass
        
        elif answer == "c":
            pass
        
        elif answer == "b":
            pass
        
        elif answer == "e":
            pass
        
        elif answer == "t":
            pass
        
#endregion
   




#region Playing functions

def simulateChessGame(p1,p2):
    #uses 2 players to play a game.
    #returns the end result as -1,0 or 1 and the final board
    board = chess.Board()
    while board.outcome() is None:
        if board.ply()%2 == 0: #White plays
            try:
                move = p1.doMove(board)
            except UnboundLocalError:
                print(board.fen())
            
        else:
            try:
                move = p2.doMove(board)
            except UnboundLocalError:
                print(board.fen())
        board.push(move)
    
    result = gameover(board)
    return result,board 

def trainBots(p1,p2,n,train_p1 = True, train_p2 = True):
    pass #Not implemented yet

def simulateMultipleGames(p1,p2,n):
    count = [0,0,0]
    for i in range(n):
        result = simulateChessGame(p1,p2)[0]
        if result == 1:
            count[0] += 1
        elif result == 0:
            count[1] += 1
        elif result == -1:
            count[2] += 1
    print(count)

#endregion





#region Player

class Player:
    def __init__(self):
        pass
    
    def doMove(self,board):
        raise NotImplementedError("Subclasses implement this method")
    
    def fit(self,library):
        #it's meant to do nothing to avoid raising unecessary errors
        pass 
    
class HumanPlayer(Player):
    def doMove(self):
        pass
    
class BotPlayer(Player):
    def __init__(self,Evaluation,search_funct,search_param):
        super().__init__()
        self.Evaluation = Evaluation
        self.search_funct = search_funct
        self.search_param = search_param
        
    def doMove(self,board):
        move = self.search_funct(board,self.search_param,self.Evaluation)
        return move
    
    def fit(self,library):
        self.Evaluation.fit(library)
    
#endregion





#region Search


def minimax_search(board,n,eval): #simple minimax        
    poss_moves = list(board.legal_moves)
    fen = board.fen()
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf
            
        if n>1: #Branches
            if board.turn:
                value = -200
                for move in poss_moves:
                    t_board = chess.Board(fen)
                    t_board.push(move)
                    t_value = minimax_search(t_board,n-1,eval)[1]
                    if t_value > value:
                        value = t_value
                        best_move = move
                    elif t_value == value:
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move
            else:
                value = 200
                for move in poss_moves:
                    t_board = chess.Board(fen)
                    t_board.push(move)
                    t_value = minimax_search(t_board,n-1,eval)[1]
                    if t_value < value:
                        value = t_value
                        best_move = move
                    elif t_value == value:
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move                              
        else: #Leafs
            list_boards = []
            for move in poss_moves:
                t_board = chess.Board(board.fen())
                t_board.push(move)
                list_boards += [t_board,]
                values = eval.evaluate(list_boards)                       
            if board.turn: #White
                m = max(values)
                moves = [i for i, j in enumerate(values) if j == m]
            else: #black
                m = min(values)
                moves = [i for i, j in enumerate(values) if j == m]         
            best_move = poss_moves[random.choice(moves)]
            
        
    else:
        if board.is_game_over():
            if board.is_checkmate():
                value = 1*(board.turn*(-2)+1)
                best_move = "checkmate?"
            elif board.is_insufficient_material():
                value = 0
                best_move = "insf material?"
            elif board.is_stalemate():
                value = 0
                best_move = "stalemate?"
        else:
            print("error type 1",fen)
            
    try:
        return best_move,value
    except:
        print("error type 2",fen)
        
def AB_prunning_search(board,n,eval,alpha=-2,beta=2): #alpha-beta prunning        
    poss_moves = list(board.legal_moves)
    evals = []
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf                                  
        if n>1: #Branch Nodes
            if board.turn: #White
                value = -5
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    t_value = AB_prunning_search(t_board,n-1,eval,alpha,beta)[1]
                    if t_value > value:
                        value = t_value
                        best_move = move
                    elif t_value == value:
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move
                    if value > beta:
                        break
                    alpha = max(alpha,value)        
                         
            else: #Black
                value = 5
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    t_value = AB_prunning_search(t_board,n-1,eval,alpha,beta)[1]
                    if t_value < value:
                        value = t_value
                        best_move = move
                    elif t_value == value: #better move
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move
                    if value < alpha: #equally good move
                        break
                    beta = min(beta,value) 
        
        elif n==1: #Leafs  
            list_boards = []
            for move in poss_moves:
                t_board = chess.Board(board.fen())
                t_board.push(move)
                list_boards += [t_board,]
                values = eval.evaluate(list_boards)                       
            if board.turn: #White
                m = max(values)
                moves = [i for i, j in enumerate(values) if j == m]
            else: #black
                m = min(values)
                moves = [i for i, j in enumerate(values) if j == m]         
            best_move = poss_moves[random.choice(moves)        ]
    else:
        if board.is_game_over():
            if board.is_checkmate():
                value = 2*(board.turn*(-2)+1)
                best_move = "checkmate?"
            elif board.is_insufficient_material():
                value = 0
                best_move = "insf material?"
            elif board.is_stalemate():
                value = 0
                best_move = "stalemate?"
        else:
            print(board.fen()) 
    return best_move



#endregion





#region Evaluation

class Evaluation:
    def __init__(self,traits,calc_funct):
        self.traits = traits
        self.calc_funct = calc_funct
        
    def evaluate(self,board):
        raise NotImplementedError("Subclasses implement this method")
    
    def fit(self,library):
        #it's meant to do nothing to avoid raising unecessary errors
        pass
    
    
#region ManualEvaluation

class ManualEvaluation(Evaluation):
    def evaluate(self, list_boards):
        values = []
        for board in list_boards:
            white = 0
            black = 0
            for trait in self.traits:
                change = trait.getValues(board)
                white += change[0]
                black += change[1]
            values += [self.calc_funct(white,black),]
        return values


#region ManualTrait
class ManualTrait:
    def __init__(self,trait_funct,trait_param):
        self.funct = trait_funct
        self.param = trait_param
        
    def getValues(self,board):
        return self.funct(board,self.param)
 
    
def piece_value(board,set):
    value_dict =[{chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:100}, # Common knowledge
                 {chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3.5,chess.ROOK:5,chess.QUEEN:10,chess.KING:100}, # Turing
                 ]
    white =0
    black =0
    for square in range(64):
        p = board.piece_at(square)
        if p is not None:
            if p.color:
                white += value_dict[set][p.piece_type]
            else:
                black += value_dict[set][p.piece_type]
    
    return white,black
 
def pawn_advancement(board,weight): #linear function that takes pawn advancement and return it's value
    #white advancement
    squares = np.array(board.pieces(chess.PAWN,chess.WHITE))
    white = weight * sum(squares//8-1)
    
    #black advancement
    squares = np.array(board.pieces(chess.PAWN,chess.BLACK))
    black = weight * sum((64-squares)//8-1)
    
    return white, black  
  

      


#endregion





#region Manual calc_funct

def hyptan(z,w):
    return((np.exp(z/w)-np.exp(-z/w))/(np.exp(z/w)+np.exp(-z/w)))

def calc_zero(board):
    return 0

def calc_sum(white,black):#most basic version but sees a sacrifice the same with or without a point difference
    return hyptan(white-black,1)

def calc_div(white,black):#takes into account the point difference but is unbalanced
    return hyptan(white/black,1)

def calc_sq_dif(white,black) :#Balanced, gives a bigger score difference when points are higher
    return hyptan(white**2-black**2,100)

def calc_sqrt_dif(white,black):
    return hyptan(white**(1/2)-black**(1/2),1)
#endregion

#endregion





#region NNEvaluation
class NNEvaluation(Evaluation):
    def evaluate(self, board):
        bitboards = np.array()
        for trait in self.traits:
            np.append(trait(board))
        results = self.calc_funct.predict(bitboards)
        return results
            
    
    def fit(self,library):
        pass #Not implemented yet
    

#region NNTrait
def position_bitboard(board): #pnbrqkPNBRQK #bW
    
    map = board.piece_map()
    piece_list = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    bitboards = {piece: np.zeros((8,8)) for piece in piece_list}
    
    for square, piece in map.items():
        bitboards[piece.symbol()][divmod(square,8)] = 1
        
    return  np.array([bitboards[piece] for piece in piece_list])
#endregion





#region NeuralNetworks
#endregion    

#endregion

#endregion





#region Misc functions

def gameover(board):
    state = board.outcome()
    if state.termination == 1:
        if state.winner:
            return 1
        else:
            return -1
    else:
        return 0
    
def load_library():
    f = open("Game_library.pgn","r")
    file = f.read()
    lines = file.split("\n")
    games = []
    state = 0
    for line in lines:
        if state == 0: #Intro
            if len(line)==0: #goes from intro to game
                state = 1
                game = []
            elif "960" in line: #goes from intro to 960
                state = 2
                substate = 1
                
        elif state == 1: #Game
            if len(line)!=0: #save the line
                game +=[line,]
            else: #end of game back to intro
                state = 0
                games +=[game,]
        
        elif state == 2: #960
            if substate == 1:#intro of 960
                if len(line) == 0: #end of intro
                    substate = 0
            elif substate == 0: #960 game
                if len(line) == 0: #end of 960 game
                    state = 0
                
    #for game in games:
    #    print(game)   
    
    ngames = []    
        
    for n,game in enumerate(games):
        ngame = []
        for line in game:
            for item in line.split(" "):
                ngame +=[item,]
        ngames += [ngame,]
     
    trainset = []
    y = []
    for ngame,game in enumerate(ngames):
        t_y = game.pop()
        board = chess.Board()
        #print(game)
        for n,item in enumerate(game):
            if n%3 != 0:
                #print(item)
                move = board.parse_san(item)
                board.push(move)
                #print(board)
                #print()
                trainset += [chess.Board(board.fen()),]
                if t_y == "1/2-1/2":
                    y += [0,]
                elif t_y == "1-0":
                    y += [1,]
                elif t_y == "0-1":
                    y += [-1,]
     
    library = np.array([trainset,y])
    return library    

#endregion





#region Preset instances





#region Lists
search_functs = [minimax_search, AB_prunning_search]
calc_functs = [calc_zero, calc_sum, calc_div, calc_sq_dif, calc_sqrt_dif]
manual_traits = [piece_value, pawn_advancement]
nn_traits = [position_bitboard]
#endregion





#region Traits
basic_trait = ManualTrait(piece_value,0)
#endregion




#region Evaluation
basic_evaluation = ManualEvaluation([basic_trait],calc_sum)
#endregion




#region Bots
basic_bot = BotPlayer(basic_evaluation,AB_prunning_search,1)

#endregion




#endregion


