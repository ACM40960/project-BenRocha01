#Project done by
#Bernardo Rocha
#
#for the Projects in Maths Modelling module in UCD




#region Imports
import chess
import chess.pgn
import random
import copy
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
    print("V 0.2.10")
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
                raise UnboundLocalError(board.fen())
            
        else:
            try:
                move = p2.doMove(board)
            except UnboundLocalError:
                raise UnboundLocalError(board.fen())
        board.push(move)
    
    result = gameover(board)
    return result,board 

def trainByPlaying(Player1,Player2,n,trainPlayer1 = True, trainPlayer2=True):
    games = []
    for i in range(n):
        result,board = simulateChessGame(Player1,Player2)
        games += [[result,board],] 
    library = gamesToLibrary(games)
    if trainPlayer1:
        Player1.fit(library[0],library[1])
    if trainPlayer2:
        Player2.fit(library[0],library[1])

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
    def __init__(self,Evaluator,Searcher):
        super().__init__()
        self.Evaluator = Evaluator
        self.Searcher = Searcher
        
    def doMove(self,board):
        move = self.Searcher.search(board,self.Evaluator)
        return move
    
    def fit(self,library):
        self.Evaluator.fit(library)
    
#endregion





#region Search

class Searcher:
    def __init__(self,funct,param):
        self.funct = funct
        self.param = param
        
    
    def search(self,board,Evaluator):
        return self.funct(board,self.param,Evaluator)[0]





def minimax(board,n,Evaluator): #simple minimax        
    poss_moves = list(board.legal_moves)
    fen = board.fen()
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf
            
        if n>1: #Branches
            if board.turn:
                value = -200
                for move in poss_moves:
                    t_board = chess.Board(fen)
                    t_board.push(move)
                    t_value = minimax(t_board,n-1,Evaluator)[1]
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
                    t_value = minimax(t_board,n-1,Evaluator)[1]
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
                values = Evaluator.evaluate(list_boards)                       
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
        
def AB_prunning(board,n,Evaluator,alpha=-2,beta=2): #alpha-beta prunning        
    poss_moves = list(board.legal_moves)
    evals = []
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf                                  
        if n>1: #Branch Nodes
            if board.turn: #White
                value = -5
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    t_value = AB_prunning(t_board,n-1,Evaluator,alpha,beta)[1]
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
                    t_value = AB_prunning(t_board,n-1,Evaluator,alpha,beta)[1]
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
                values = Evaluator.evaluate(list_boards)                       
            if board.turn: #White
                value = max(values)
                moves = [i for i, j in enumerate(values) if j == value]
            else: #black
                value = min(values)
                moves = [i for i, j in enumerate(values) if j == value]         
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
    return best_move,value

def ID_AB_prunning(board,n,Evaluator,alpha=-2,beta=2): #alpha-beta prunning        
    poss_moves = list(board.legal_moves)
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf 
        
        list_boards = []
        
        for move in poss_moves:
            t_board = chess.Board(board.fen())
            t_board.push(move)
            list_boards += [t_board,]
            values = Evaluator.evaluate(list_boards)
        
        sorted_indices = sorted(range(len(values)),key=lambda k: values[k])
        values = [values[i] for i in sorted_indices]
        poss_moves = [poss_moves[i] for i in sorted_indices]
        
        
                                         
        if n>1: #Branch Nodes
            if board.turn: #White
                value = -5
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    t_value = AB_prunning(t_board,n-1,Evaluator,alpha,beta)[1]
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
                    t_value = AB_prunning(t_board,n-1,Evaluator,alpha,beta)[1]
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
            if board.turn: #White
                value = max(values)
                moves = [i for i, j in enumerate(values) if j == value]
            else: #black
                value = min(values)
                moves = [i for i, j in enumerate(values) if j == value]         
            best_move = poss_moves[random.choice(moves)]
        
        
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
    return best_move,value


#endregion





#region Evaluation

class Evaluator:
    def __init__(self,traits,model):
        self.traits = traits
        self.model = model
        
    def evaluate(self,board):
        raise NotImplementedError("Subclasses implement this method")
    
    def fit(self,library):
        #it's meant to do nothing to avoid raising unecessary errors
        pass
    
    
#region ManualEvaluation

class ManualEvaluator(Evaluator):
    def evaluate(self, list_boards):
        values = []
        for board in list_boards:
            white = 0
            black = 0
            for trait in self.traits:
                change = trait.getValues(board)
                white += change[0]
                black += change[1]
            values += [self.model(white,black),]
        return values



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
class NNEvaluator(Evaluator):
    def evaluate(self, board):
        bitboards = np.array()
        for trait in self.traits:
            np.append(trait(board))
        results = self.calc_funct.predict(bitboards)
        return results
            
    
    def fit(self,library):
        pass #Not implemented yet
    



#region Model handling
class Model(tf.keras.models.Sequential):
    def setName(self,name):
        self.costum_name = name
        
    def getName(self):
        return self.costum_name
    
    def save(self):
        if hasattr(self, 'name'):
            super().save("model/"+self.costum_name + ".keras")
        else:
            raise ValueError("Model name has not been set. Use setName() method to set the model name.")


#endregion





#region NeuralNetworks
#endregion    

#endregion

#endregion





#region Trait
class Trait:
    #Missing description method/attribute
    def __init__(self,name,funct):
        self.name = name
        self.funct = funct
    
    def getValues(self,board):
        raise NotImplementedError("This is meant ot be implemented by the subclasses")


#region ManualTrait

class ManualTrait(Trait):        
    def set_param(self,param):
        self.param = param
        
    def getValues(self,board):
        return self.funct(board,self.param)
 
    
def piece_value_funct(board,set):
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

piece_value = ManualTrait("Piece value", piece_value_funct)


def pawn_advancement_funct(board,weight): #linear function that takes pawn advancement and return it's value
    #white advancement
    squares = np.array(board.pieces(chess.PAWN,chess.WHITE))
    white = weight * sum(squares//8-1)
    
    #black advancement
    squares = np.array(board.pieces(chess.PAWN,chess.BLACK))
    black = weight * sum((64-squares)//8-1)
    
    return white, black  
 
pawn_advancement = ManualTrait("Pawn advancement",pawn_advancement_funct) 

      


#endregion





#region NNTrait

class NNTrait(Trait):
    def getValues(self, board):
        return self.funct(board)
    



def position_bitboard_funct(board): #pnbrqkPNBRQK #bW
    
    map = board.piece_map()
    piece_list = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    bitboards = {piece: np.zeros((8,8)) for piece in piece_list}
    
    for square, piece in map.items():
        bitboards[piece.symbol()][divmod(square,8)] = 1
        
    return  np.array([bitboards[piece] for piece in piece_list])

position_bitboard = NNTrait("Positions bitboard",position_bitboard_funct)
#endregion
#endregion





#region Lists
search_functs = [minimax, AB_prunning]
calc_functs = [calc_zero, calc_sum, calc_div, calc_sq_dif, calc_sqrt_dif]
manual_traits = [piece_value, pawn_advancement]
nn_traits = [position_bitboard]
#endregion





#region Builders not actual builders


        
def TraitBuilder():
    answer = None
    while answer != "q":
        print("""Trait Builder options:
              -Manual Trait (m)
              -NN Trait(n)
              -quit(q)""")
        answer = input("answer: ")
        if answer == "m":
            trait = ManualTraitBuilder()
            answer = "q"
        elif answer == "n":
            trait = NNTraitBuilder()
            answer = "q"
        elif answer !="q":
            print("invalid input")

def ManualTraitBuilder():
    answer = None
    while answer != "q":
        print("""Manual Trait Builder:
              choose a function:""")
        options = []
        for n,i in manual_traits:
            print("-"+i.name+"("+str(n)+")")
            options += [n,]
            
        try:
            trait_n = int(input("answer: "))
            if trait_n in options:
                trait = manual_traits[trait_n]
                print("Set a parameter value:")
                while answer != "q":
                    try:
                        param = int(input("answer"))
                        trait.set_param(param)
                        answer = "q"
                    except ValueError:
                        print("Invalid input. Please enter a number")
            else:
                print("Option out of bounds")
        except ValueError:
            print("Invalid input. Please enter a number")
            
    return trait

def NNTraitBuilder():
    answer = None
    while answer != "q":
        print("""Manual Trait Builder:
              choose a function:""")
        options = []
        for n,i in manual_traits:
            print("-"+i.name+"("+str(n)+")")
            options += [n,]
            
        try:
            trait_n = int(input("answer: "))
            if trait_n in options:
                trait = manual_traits[trait_n]
                print("Set a parameter value:")
                answer = "q"
            else:
                print("Option out of bounds")
        except ValueError:
            print("Invalid input. Please enter a number")
    return trait

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
    
def loadFromRepository(): #with gamesToLibrary takes close to 
    pgn = open("libraries/Game_library.pgn")
    game = 0
    games = []
    game = chess.pgn.read_game(pgn)
    while game is not None:
        games += [[game.headers["Result"],game.end().board()],]
        game = chess.pgn.read_game(pgn)
    pgn.close()
    return games
   

def gamesToLibrary(o_games,minply = 0):
    games = copy.deepcopy(o_games)
    library = []
    for game in games:
        if game[0]=="1-0" or game[0] == 1:
            result = 1
        elif game[0]=="1/2-1/2" or game[0] == 0:
            result = 0
        else:
            result = -1
        while game[1].ply() > minply:
            
            library += [[chess.Board(game[1].fen()),result],]
            game[1].pop()
            
    library = np.array(library).T
    return library

#endregion





#region Preset instances








#region Traits
basic_trait = piece_value
basic_trait.set_param(0)
#endregion





#region Searcher
ab_searcher_2 = Searcher(AB_prunning,2)
#endregion





#region Evaluation
basic_evaluation = ManualEvaluator([basic_trait],calc_sum)
#endregion




#region Bots
basic_bot = BotPlayer(basic_evaluation,ab_searcher_2)

#endregion




#endregion


