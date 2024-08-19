#Project done by
#Bernardo Rocha
#
#for the Projects in Maths Modelling module in UCD




#region Imports
import chess
import chess.pgn
import chess.svg
from IPython.display import SVG, display
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


  


#region Playing functions

def simulateChessGame(p1,p2):
    #uses 2 players to play a game.
    #returns the end result as -1,0 or 1 and the final board
    board = chess.Board()
    while board.outcome() is None:
        if board.ply()%2 == 0: #White plays
            try:
                move = p1.doMove(board=board)
            except UnboundLocalError:
                raise UnboundLocalError(board.fen())
            
        else:
            try:
                move = p2.doMove(board)
            except UnboundLocalError:
                raise UnboundLocalError(board.fen(),)
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
    def doMove(self,board):
        display(SVG(chess.svg.board(board,size =400)))
        poss_moves = list(board.legal_moves)
        #print(poss_moves)
        invalid = True
        while invalid:
            try:
                move = chess.Move.from_uci(input("your Move?"))
                if move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    display(SVG(chess.svg.board(t_board,size =400)))
                    return move
                else:
                    print("Not a valid Move")
            except chess.InvalidMoveError:
                print("Not a Move, try again")
    
class BotPlayer(Player):
    def __init__(self,name,Evaluator,Searcher):
        self.name = name
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
    def __init__(self,name,descr,funct,depth,param):
        self.name = name
        self.descr = descr
        self.funct = funct
        self.depth = depth
        self.param = param
        
    
    def search(self,board,Evaluator):
        return self.funct(board,self.depth,self.param,Evaluator)[0]


class SearcherDirector:
    @staticmethod
    def minimax(depth):
        descr = None
        return Searcher("Minimax, depth "+str(depth),descr,minimax,depth,None)
    
    @staticmethod
    def minimax_NN(depth):
        descr = None
        return Searcher("Minimax, depth "+str(depth),descr,minimax_NN,depth,None)
    
    @staticmethod
    def AB_pruning(depth):
        descr = None
        return Searcher("Alpha-Beta prunning, depth "+str(depth),descr,AB_pruning,depth,None)
    
    @staticmethod
    def AB_pruning_NN(depth):
        descr = None
        return Searcher("Alpha-Beta prunning, depth "+str(depth),descr,AB_pruning_NN,depth,None)
    
    @staticmethod
    def ID_AB_pruning(depth):
        descr = None
        return Searcher("Iterative deepening Alpha-Beta prunning, depth "+str(depth),descr,ID_AB_pruning,depth,None)
    




def minimax(board,depth,param,Evaluator):
    if type(depth)!=int:
        raise TypeError("minimax parameter needs to be integer")
    best_move=0       
    poss_moves = list(board.legal_moves)
    first_move = True
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf                                  
        for move in poss_moves:
            t_board = chess.Board(board.fen())
            t_board.push(move)
            if depth > 1:
                value = minimax(t_board,depth-1,param,Evaluator)
            elif depth == 1:
                value = Evaluator.evaluate([t_board])
            if board.turn: #White turn
                if first_move:
                    best_value = value
                    best_move = move
                if value > best_value:
                    best_value = value
                    best_move = move
            else: #Black turn
                if first_move:
                    best_value = value
                    best_move = move
                if value < best_value:
                    best_value = value
                    best_move = move
        
    else: #No more moves
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
    return best_move,best_value

def minimax_NN(board,depth,param,Evaluator,master = True):
    if type(depth)!=int:
        raise TypeError("minimax parameter needs to be integer")
    best_move=0       
    poss_moves = list(board.legal_moves)
    list_boards = []
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf                                  
        for move in poss_moves:
            t_board = chess.Board(board.fen())
            t_board.push(move)
            if depth > 1:
                list_boards += [minimax_NN(t_board,depth-1,param,Evaluator,False)[1],]
            else:
                list_boards += [t_board,]
        if master:
            def flatten(nested_list,layer=0):
                flat_list = []
    
                for n,item in enumerate(nested_list):
                    if type(item) == list:
                        item = flatten(item,layer+1)
                        flat_list.extend(item)
                    else:
                        flat_list.append(item)
                return flat_list
            
            def renest(nested):
                nonlocal n
                n_list = []
                for item in nested:
                    if type(item) == list:
                        n_list += [renest(item),]
                    else:
                        n_list += [float(values[n]),]
                        n +=1
                return n_list 
            
            def minimax_nested(nested,turn):
                values = []
                if type(nested) is list:
                    if len(nested) == 0:
                        return None
                    for i in nested:
                        if type(i) is list:
                            add = minimax_nested(i, not turn)
                        else:
                             add = float(i)
                        if add != None:
                            values += [add,]
                else:
                    values = nested
                #print("new line")
                #print(values)
                if turn:
                    return max(values)
                else:
                    return min(values)
                         
            flat_list = flatten(list_boards)
            boards = flat_list
            values = Evaluator.evaluate(boards)
            n = 0
            nested = renest(list_boards)
            
            if board.turn:
                best_value = -math.inf
            else:
                best_value = math.inf
            
            for n,i in enumerate(nested):
                value = minimax_nested(i,not board.turn)
                if board.turn:
                    if value > best_value:
                       value = best_value
                       best_move = poss_moves[n]
                else:
                    if value < best_value:
                       value = best_value
                       best_move = poss_moves[n]            
        
    else: #No more moves
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
    return best_move,list_boards
        
def AB_pruning(board,depth,param,Evaluator,alpha=-math.inf,beta=math.inf): #alpha-beta prunning   
    
    if type(depth)!=int:
        raise TypeError("AB_prunning parameter needs to be integer")
    best_move = 0     
    poss_moves = list(board.legal_moves)
    
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf                                  
    
        for move in poss_moves:
            t_board = chess.Board(board.fen())
            t_board.push(move)
            if depth>1: #Branch Nodes
                value = AB_pruning(t_board,depth-1,param,Evaluator,alpha,beta)[1]
            elif depth == 1 : #Leaf Nodes
                board_list = [t_board]
                value = Evaluator.evaluate(board_list)[0]
            if board.turn: #White turn
                if value > alpha:
                    alpha = value
                    best_move = move
                if value > beta:
                    break
            else: #Black turn
                if value < beta:
                    beta = value
                    best_move = move
                if value < alpha:
                    break

    else: #No more moves
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

def AB_pruning_NN(board,depth,param,Evaluator,alpha=-math.inf,beta=math.inf): #alpha-beta prunning   
    if type(depth)!=int:
        raise TypeError("AB_prunning parameter needs to be integer")
    best_move = 0     
    poss_moves = list(board.legal_moves)
    
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf                                  
        
        if depth>1: #Branch Nodes
            for move in poss_moves:
                t_board = chess.Board(board.fen())
                t_board.push(move)
                value = AB_pruning_NN(t_board,depth-1,param,Evaluator,alpha,beta)[1]
                if board.turn: #White turn
                    if value > alpha:
                        alpha = value
                        best_move = move
                    if value > beta:
                        break
                else: #Black turn
                    value = value
                    if value < beta:
                        beta = value
                        best_move = move
                    if value < alpha:
                        break
                
        elif depth == 1: #Leaf Nodes
            board_list = []
            for move in poss_moves:
                t_board = chess.Board(board.fen())
                t_board.push(move)
                board_list+=[t_board,]
            value = Evaluator.evaluate(board_list)
            if board.turn: #White turn
                value = np.max(value)
            else: #Black turn
                value = np.min(value)
            

    else: #No more moves
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

def ID_AB_pruning(board,depth,param,Evaluator,alpha=-math.inf,beta=math.inf): #alpha-beta prunning   
    if type(depth)!=int:
        raise TypeError("AB_prunning parameter needs to be integer")
    best_move = 0     
    poss_moves = list(board.legal_moves)
    
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf                                  
        board_list = []
        #getting the boards
        for move in poss_moves:
            t_board = chess.Board(board.fen())
            t_board.push(move)
            board_list +=[t_board]
            
        values = Evaluator.evaluate(board_list)   
        
        #sorting the boards
        t_combined = list(zip(values,board_list,poss_moves))
        combined = sorted(t_combined,key=lambda x:x[0])
        sorted_boards = [m for _,m,_ in combined]
        sorted_moves = [m for _, _,m in combined]
        if depth>1: #Branch Nodes
            for board in sorted_boards:
                value = AB_pruning(board,depth-1,param,Evaluator,alpha,beta)[1]
                if board.turn: #White turn
                    if value > alpha:
                        alpha = value
                        best_move = sorted_moves[-1]
                    if value > beta:
                        break
                    else: #Black turn
                        value = min(values)
                        if value < beta:
                            beta = value
                            best_move = sorted_moves[0]
                        if value < alpha:
                            break   
                                        
        elif depth == 1: #Leaf Nodes
            if board.turn: #White turn
                value = values[-1]
                best_move = sorted_moves[-1]
            else: #Black turn
                value = values[0]
                best_move = sorted_moves[0]
                

    else: #No more moves
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
    def __init__(self,name,descr,features,model):
        self.name = name
        self.descr = descr
        self.features = features
        self.model = model
        
    def evaluate(self,board):
        raise NotImplementedError("Subclasses implement this method")
    
    def fit(self,library):
        #it's meant to do nothing to avoid raising unecessary errors
        pass


class EvaluatorDirector:
    @staticmethod
    def random():
        descr = None
        return ManualEvaluator("Random",descr,None,calc_zero)
    
    @staticmethod
    def sum(features):
        descr = None
        return ManualEvaluator("Sum",descr,features,calc_sum)
    
    @staticmethod
    def div(features):
        descr = None
        return ManualEvaluator("Divison",descr,features,calc_div)
    
    @staticmethod
    def sq_dif(features):
        descr = None
        return ManualEvaluator("Square difference",descr,features,calc_sq_dif)
    
    
    @staticmethod
    def sqrt_dif(features):
        descr = None
        return ManualEvaluator("Square root difference",descr,features,calc_sqrt_dif)
    
    @staticmethod
    def NN(name,minply):
        descr = None
        features = [FeatureDirector.PositionBitboard()]
        model = tf.keras.saving.load_model(f"model\{name}_{minply}.keras")
        return NNEvaluator(f"NN {name}_{minply}",descr,features,model)
 
 
NNnames=["Single256","Single128","Single64","Single32","Pair256","Double128","Single256_Double256_128","Single128_Double128"]
NNminply = ["0","15","30","45","60"]

def loadModels():
    NNModels = {}
    for name in NNnames:
        for minply in NNminply:
            NNModels[name+"_"+minply]=EvaluatorDirector.NN(name,minply)
    return NNModels
            
    

#region ManualEvaluation

class ManualEvaluator(Evaluator):
    def evaluate(self, list_boards):
        values = []
        for board in list_boards:
            white = 0
            black = 0
            for feature in self.features:
                change = feature.getValues(board)
                white += change[0]
                black += change[1]
            values += [self.model(white,black),]
        
        return values



#region Manual calc_funct

def hyptan(z,w):
    return((np.exp(z/w)-np.exp(-z/w))/(np.exp(z/w)+np.exp(-z/w)))

def calc_zero(white,black):
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
    def evaluate(self, list_boards):
        values = []
        n_boards = 0
        for feature in self.features:
            n_boards += feature.n_bitboards
        for board in list_boards:
            bitboards = np.zeros((n_boards,8,8))
            counter = 0
            for n,feature in enumerate(self.features):
                bitboards[counter:counter+feature.n_bitboards] = feature.getValues(board)
                counter += feature.n_bitboards+1
            values += [bitboards,]
        values = np.array(values)       
        results = self.model.predict(values,verbose = 0)
        return results
            
    
    def fit(self,library):
        pass #Not implemented yet
    



#region Model handling
class Model(tf.keras.models.Sequential):
    
    def save(self):
        if hasattr(self, 'name'):
            super().save("model/"+self.name + ".keras")
        else:
            raise ValueError("Model name has not been set. Use setName() method to set the model name.")






#endregion





#region NeuralNetworks
#endregion    

#endregion

#endregion





#region Feature
class Feature:
    #Missing description method/attribute
    def __init__(self,name,descr,funct):
        self.name = name
        self.funct = funct
    
    def getValues(self,board):
        raise NotImplementedError("This is meant ot be implemented by the subclasses")



        
class FeatureDirector:
     
    #manual functions   
    @staticmethod 
    def PieceValue(param):
        descr = None
        return ManualFeature("Piece Value"+str(param),descr,piece_value_funct,param)
    
    @staticmethod
    def PawnAdvancement(param):
        descr = None
        return ManualFeature("Pawn Advancement"+str(param),descr,pawn_advancement_funct,param)
    
    #nn functions
    @staticmethod
    def PositionBitboard():
        descr = None
        return NNFeature("Position Bitboard",descr,position_bitboard_funct,12)
    
#region ManualFeature

class ManualFeature(Feature):
    def __init__(self,name,descr,function,param):
        super().__init__(name,descr,function)
        self.param = param
    
            
    def set_param(self,param):
        self.param = param
        
    def getValues(self,board):
        return self.funct(board,self.param)
 
    
def piece_value_funct(board,set):
    value_dict =[{chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:0}, # Common knowledge
                 {chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3.5,chess.ROOK:5,chess.QUEEN:10,chess.KING:0}, # Turing
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



def pawn_advancement_funct(board,weight): #linear function that takes pawn advancement and return it's value
    #white advancement
    white = 0
    squares = list(board.pieces(chess.PAWN,chess.WHITE))
    for i in squares:
        white = weight * (i//8-1)
    
    #black advancement
    black = 0
    squares = list(board.pieces(chess.PAWN,chess.BLACK))
    for i in squares:
        black = weight * (i//8-1)
    
    return white, black  
 

      


#endregion





#region NNFeature

class NNFeature(Feature):
    def __init__(self,name,descr,funct,n_bitboards):
        super().__init__(name,descr,funct)
        self.n_bitboards = n_bitboards
    
    def getValues(self, board):
        return self.funct(board)
    



def position_bitboard_funct(board): #pnbrqkPNBRQK #bW
    
    map = board.piece_map()
    piece_list = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
    bitboards = {piece: np.zeros((8,8)) for piece in piece_list}
    
    for square, piece in map.items():
        bitboards[piece.symbol()][divmod(square,8)] = 1
        
    return  np.array([bitboards[piece] for piece in piece_list])


#endregion
#endregion



#region Misc functions

def gameover(board):
    state = board.outcome()
    if state.termination == chess.Termination.CHECKMATE:
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


def flatten(nested_list, depth):
    flat_list = []
    coordinates = []
    def helper(sublist, current_coords, level):
        if level == depth - 1:  # If we're at the second-to-last level
            for i, item in enumerate(sublist):
                flat_list.append(item)
                coordinates.append(current_coords + [i])
        else:
            for i, item in enumerate(sublist):
                helper(item, current_coords + [i], level + 1)
    
    helper(nested_list, [], 0)
    return flat_list,coordinates

#endregion





#region Preset instances



#endregion


