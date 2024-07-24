import chess
import random
import math
import time
import numpy as np
import tensorflow as tf

#Global variables
board = chess.Board()
library = 0
Bots = []

#Main functions
def version():
    print("V 0.1.9")
    ###Changelog
    # V 0.1.0   -The whole evaluation system was revamped to be modular
    #           -

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

def botPlay(branch1,n1,eval1,branch2,n2,eval2,n,progress=False):
    itime = time.time()
    wins = 0 
    draws = 0
    losses = 0
    for game in range(n):
        board = chess.Board()
        
        if game%2 == 0:
            while board.outcome() is None:
                if board.ply()%2 == 0:
                    try:
                        move = branch1(board,n1,eval1)[0] #
                    except UnboundLocalError:
                        print(board.fen())
                    board.push(move)
                
                else:
                    try:
                        move = branch2(board,n2,eval2)[0]
                    except UnboundLocalError:
                        print(board.fen())
                    board.push(move)
            if board.outcome().winner:
                wins += 1
            elif board.outcome().winner is None:
                draws += 1
            else:
                losses += 1
        else:
            while board.outcome() is None:
                if board.ply()%2 == 0:
                    try:
                        move = branch2(board,n2,eval2)[0]
                    except UnboundLocalError:
                        print(board.fen())
                    board.push(move)
                
                else:
                    try:
                        move = branch1(board,n1,eval1)[0]
                    except UnboundLocalError:
                        print(board.fen())
                    board.push(move)
            if not board.outcome().winner:
                wins += 1
            elif board.outcome().winner is None:
                draws += 1
            else:
                losses += 1
        if progress:
            print(game)
    ftime= time.time()
    ttime = ftime-itime       
    return wins,draws,losses,ttime

def finalBotPlay(branch1,n1,eval1,branch2,n2,eval2,n,progress=False):
    wins = 0 
    draws = 0
    losses = 0
    for game in range(n):
        board = chess.Board()
        
        if game%2 == 0:
            while board.outcome() is None:
                if board.ply()%2 == 0:
                    move = branch1(board,n1,eval1)[0]
                    board.push(move)
                
                else:
                    move = branch2(board,n2,eval2)[0]
                    board.push(move)
            if board.outcome().winner:
                wins += 1
            elif board.outcome().winner is None:
                draws += 1
            else:
                losses += 1
        else:
            while board.outcome() is None:
                if board.ply()%2 == 0:
                    move = branch2(board,n2,eval2)[0]
                    board.push(move)
                
                else:
                    move = branch1(board,n1,eval1)[0]
                    board.push(move)
            if not board.outcome().winner:
                wins += 1
            elif board.outcome().winner is None:
                draws += 1
            else:
                losses += 1
        if progress:
            print(game)        
    return wins,draws,losses

class Evaluation:
    def __init__(self,manual,traits):
        self.manual = manual #If the evaluation is manual or ML - boolean
        self.traits = traits # List of the trait functions to be used
        


class Bot:
    def __init__(self,search_funct,Evaluation):
        self.search = search_funct #This is the search function
        self.evaluation = Evaluation #This is the Evaluation object
        
    
        


#Bots

def Player(board,n,eval):
    print(board)
    moves = list(board.legal_moves)
    lm = [x.uci() for x in moves]
    print(moves_by_piece(board))
    m = input("Enter your move (e.g. b4e4):")
    move = chess.Move.from_uci(m)
    if move in board.legal_moves:
        return(move)
    else:
        print("Invalid move")

def Halfred(board,n,eval): #only predicts half a turn

    vector =(1,0)
    fen = board.fen()
    moves = list(board.legal_moves)
    evals = []
    for move in moves:
        t_board = chess.Board(fen)
        t_board.push(move)
        evals += [sum_eval(t_board,vector),]
    if board.turn:
        M = max(evals)
    else:
        M = min(evals)
    good_moves = []
    for num, value in enumerate(evals):
        if value == M:
            good_moves += [num,]
    
    move = random.randrange(len(good_moves))
    
    return moves[good_moves[move]],0

def Halfred2(board,n,eval): 
    def s_branch(board,n,vector):        
        poss_moves = list(board.legal_moves)
        evals = []
        if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf
            
            if n==1: #Leafs
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    evals += [sum_eval(t_board,vector),]                  
            else: #Branch Nodes
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    evals += [s_branch(t_board,n-1)[1],]
            
            #chooses a move from the nodes        
            if board.turn:
                M = max(evals)
            else:
                M = min(evals)
            good_moves = []
            for num,value in enumerate(evals):
                if value == M:
                    good_moves += [num,]
            move = random.randrange(len(good_moves))   
            best_move = poss_moves[good_moves[move]]
        
        else:
            if board.is_game_over():
                if board.is_checkmate():
                    M = 2*(board.turn*(-2)+1)
                    best_move = "checkmate?"
                elif board.is_insufficient_material():
                    M = 0
                    best_move = "insf material?"
                elif board.is_stalemate():
                    M = 0
                    best_move = "stalemate?"
            else:
                print(board.fen())
            
            
        return best_move,M
  
    n = 2
    vector =(1,0)
    move = s_branch(board,n,vector)
    return move

def Randall(board,n,eval):
    moves = list(board.legal_moves)
    move = random.randrange(len(moves))
    return moves[move],0


#Searching functions

def search_functs():
    l=[minimax_branch,alphabeta_branch]
    lg=["Minimax","Alpha-beta pruning"]
    d={}
    for num, fun in enumerate(l):
        d[l]=lg[num]
    return d

def minimax_branch(board,n,eval): #simple minimax        
    poss_moves = list(board.legal_moves)
    fen = board.fen()
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf
            
        if n==1: #Leafs
            if board.turn:
                value = -200
                for move in poss_moves:
                    t_board = chess.Board(fen)
                    t_board.push(move)
                    t_value = eval(t_board)
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
                    t_value = eval(t_board)
                    if t_value < value:
                        value = t_value
                        best_move = move
                    elif t_value == value:
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move                  
        else: #Branch Nodes
            if board.turn:
                value = -200
                for move in poss_moves:
                    t_board = chess.Board(fen)
                    t_board.push(move)
                    t_value = minimax_branch(t_board,n-1,eval)[1]
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
                    t_value = minimax_branch(t_board,n-1,eval)[1]
                    if t_value < value:
                        value = t_value
                        best_move = move
                    elif t_value == value:
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move
        
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
        
def alphabeta_branch(board,n,eval,alpha=-2,beta=2): #alpha-beta prunning        
    poss_moves = list(board.legal_moves)
    evals = []
    if len(poss_moves)>0: #Has possible moves -> evals each Node/Leaf    
                                        
        if n>1: #Branch Nodes
            if board.turn: #White
                value = -5
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    t_value = alphabeta_branch(t_board,n-1,eval,alpha,beta)[1]
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
                    t_value = alphabeta_branch(t_board,n-1,eval,alpha,beta)[1]
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
                value = -5
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    t_value = eval(t_board)
                    if t_value > value: #better move
                        value = t_value
                        best_move = move
                    elif t_value == value: #equally good move
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move
            else: #black
                value = 5
                for move in poss_moves:
                    t_board = chess.Board(board.fen())
                    t_board.push(move)
                    t_value = eval(t_board)#sends a board receives a number
                    if t_value < value:
                        value = t_value
                        best_move = move
                    elif t_value == value:
                        r = random.random()
                        if r > 0.1:
                            value = t_value
                            best_move = move            
                    
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

#Evaluation functions

def eval_functs():
    l=[zero_eval,sum_eval,div_eval,sq_dif_eval,sqrt_dif_eval]
    lg =["zero","difference","division", "square difference","square root difference"]
    d={}
    for num, fun in enumerate(l):
        d[l]=lg[num]
    return d

def zero_eval(board):
    return 0
 
def sum_eval(board,vector):#most basic version but sees a sacrifice the same with or without a point difference
    if board.is_checkmate():
        return 2*(board.turn*(-2)+1)
    elif board.is_insufficient_material():
        return 0
    elif board.is_stalemate():
        return 0
    white,black = eval(board,vector)
    return hyptan(white-black,1)

def div_eval(board,vector):#takes into account the point difference but is unbalanced
    if board.is_checkmate():
        return 2*(board.turn*(-2)+1)
    elif board.is_insufficient_material():
        return 0
    elif board.is_stalemate():
        return 0
    white,black = eval(board,vector)
    return hyptan(white/black,1)

def sq_dif_eval(board,vector): #Balanced, gives a bigger score difference when points are higher
    if board.is_checkmate():
        return 2*(board.turn*(-2)+1)
    elif board.is_insufficient_material():
        return 0
    elif board.is_stalemate():
        return 0
    white,black = eval(board,vector)
    return hyptan(white**2-black**2,100)

def sqrt_dif_eval(board,vector): #Balanced, gives a bigger score difference points are lower. I think this is the best
    if board.is_checkmate():
        return 2*(board.turn*(-2)+1)
    elif board.is_insufficient_material():
        return 0
    elif board.is_stalemate():
        return 0
    white,black = eval(board,vector)
    return hyptan(white**(1/2)-black**(1/2),1)
 
def eval(board,vector):
    white = 0
    black = 0
    if vector[0] != 0: #piece value
        add = piece_value(board,vector[0])
        white += add[0]
        black += add[1]
    
    if vector[1] != 0: #pawns
        if "advancement" in vector[1].keys():
            add = pawn_advancement(board,vector[1]["advancement"])
            white += add[0]
            black += add[1]
        #if "doubled" in vector[1].keys():
            
    #if vector[2] != 0: #pawn structure
    #    if "d" in vector[2]: #doubled
    
    
        
            
        
        
    return white,black
        
def nn_eval(board,model):
    a = np.array([position_bitboard(board)])
    return model.predict(a)

#Value functions

def value_sets():
    d ={1:"1,3,3,5,9",2:"1,3,3.5,5,10"}
    return

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
   
#Info for NN

def position_bitboard(board): #pnbrqkPNBRQK #bW
    bitboards = np.zeros((12,8,8))
    for sq in range(64):
        p = board.piece_at(sq)
        if p is None:
            pass
        else:
            b = (p.piece_type-1)*64+(p.color*64*6)+sq
            bitboards[p.piece_type-1+p.color*6][sq//8][sq%8] = 1
    return bitboards

#Misc

def moves_by_piece(board,bot=True):
    moves = list(board.legal_moves)
    moves_dict = {}
    if bot:
        for move in moves:
            piece = board.piece_at(move.from_square)
            key = (piece.piece_type)
            try:
                moves_dict[key] += [move,]
            except KeyError:
                moves_dict[key] = [move,]
    else:
        for move in moves:
            piece = board.piece_at(move.from_square)
            key = str(piece)+" at "+str(chess.SQUARE_NAMES[move.from_square])
            try:
                moves_dict[key] += [move.uci(),]
            except KeyError:
                moves_dict[key] = [move.uci(),]
    return(moves_dict)

def hyptan(z,w):
    return((np.exp(z/w)-np.exp(-z/w))/(np.exp(z/w)+np.exp(-z/w)))

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