import chess
board = chess.Board()
board = chess.Board()
board.push(chess.Move.from_uci("f2f3"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g2g4"))
pre_mate = board.fen()
board.push(chess.Move.from_uci("d8h4"))

board
def Game(P1,P2,n):
    wins = 0 
    draws = 0
    losses = 0
    for game in range(n):
        board = chess.Board()
        c = 0
        while not board.outcome():
            if board.ply()%2 == 0:
                move = P1(board)
                board.push(move)
                
            else:
                move = P2(board)
                board.push(move)
        if board.outcome().winner:
            wins += 1
        elif board.outcome().winner is None:
            draws += 1
        else:
            losses += 1
    return wins,draws,losses
            
def Player(board):
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
    
    
import random
def Randy(board):
    moves = list(board.legal_moves)
    move = random.randrange(len(moves))
    return moves[move]


board = chess.Board()
board.push(Randy(board))
board


print(Game(Randy,Randy,1000))

def moves_by_piece(board,bot):
    moves = list(board.legal_moves)
    moves_dict = {}
    if bot:
        for move in moves:
            piece = board.piece_at(move.from_square)
            key = str(chess.SQUARE_NAMES[move.from_square])
            try:
                moves_dict[key] += [move.uci(),]
            except KeyError:
                moves_dict[key] = [move.uci(),]
    else:
        for move in moves:
            piece = board.piece_at(move.from_square)
            key = str(piece)+" at "+str(chess.SQUARE_NAMES[move.from_square])
            try:
                moves_dict[key] += [move.uci(),]
            except KeyError:
                moves_dict[key] = [move.uci(),]
    return(moves_dict)

def piece_value(piece):
    if piece.piece_type == 1: #Pawn
        return 1
    elif piece.piece_type == 2: #Knight
        return 3
    elif piece.piece_type == 3: #Bishop
        return 3
    elif piece.piece_type == 4: #Rook
        return 5
    elif piece.piece_type == 5: #queen
        return 9
    elif piece.piece_type == 6: #King
        return 0
    else:
        return 0


import random
import math
def BalancedRandy(board): #uses softmax
    moves = list(board.legal_moves)
    sum = 0
    prob_vect = []
    for move in moves:
        sum += math.exp(piece_value(board.piece_at(move.from_square)))
        prob_vect +=[sum,]
    
    choice = random.random()*sum
    
    for i in range(len(prob_vect)):
        if choice < prob_vect[i]:
            return moves[i-1]


Game(BalancedRandy,Randy,1000)

def IBalancedRandy(board): #uses softmax
    moves = list(board.legal_moves)
    sum = 0
    prob_vect = []
    for move in moves:
        sum += math.exp(-piece_value(board.piece_at(move.from_square)))
        prob_vect +=[sum,]
    
    choice = random.random()*sum
    
    for i in range(len(prob_vect)):
        if choice < prob_vect[i]:
            return moves[i-1]
        
Game(IBalancedRandy,Randy,1000)


def base_eval(board):
    eval = 0
    if board.is_checkmate():
        return 10000*(board.turn*(-2)+1)
        
    for square in range(64):
        color = board.color_at(square)
        if color is not None:
            eval += board.piece_type_at(square)*(color*2-1)
    return eval


def Halfred(board): #only predicts half a turn

    fen = board.fen()
    moves = list(board.legal_moves)
    evals = []
    for move in moves:
        t_board = chess.Board(fen)
        t_board.push(move)
        evals += [base_eval(t_board),]
    if board.turn:
        M = max(evals)
    else:
        M = min(evals)
    good_moves = []
    for num, value in enumerate(evals):
        if value == M:
            good_moves += [num,]
    
    move = random.randrange(len(good_moves))
    
    return moves[good_moves[move]]
    
    
Game(Halfred,Randy,100)