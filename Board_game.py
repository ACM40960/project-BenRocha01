
class Bitboard:
    def __init__(self,piece=None):
        """
        Every square with a piece is a one in the binary
        number that starts in h1 and runs line by line
        for example the number 576460855785291776
        represents the 4 central squares
        """
        if piece == None:
            self.board = 0            
        #White is uppercase
        if piece == "P":
            self.board = 65280
        if piece == "N":
            self.board = 66
        if piece == "B":
            self.board = 36
        if piece == "R":
            self.board = 129
        if piece == "Q":
            self.board = 16
        if piece == "K":
            self.board = 8
        #Black is lowercase
        if piece == "p":
            self.board = 71776119061217280
        if piece == "n":
            self.board = 4755801206503243776
        if piece == "b":
            self.board = 2594073385365405696
        if piece == "r":
            self.board = 9295429630892703744
        if piece == "q":
            self.board = 1152921504606846976
        if piece == "k":
            self.board = 576460752303423488
    
    def __str__(self):
        
        b = bin(self.board)
        b = b[2:]
        b = "0"*(64-len(b))+b
        f = ""
        for i in range(len(b)):
            f += b[i]
            if (i+1)%8 == 0:
                f += "\n"
        return(f)
    
    def move(self,move): 
        """Returns the new position bitboard after a move.
        Move should be defined as "(cr,CR)",
        c being the previous column and C the new as a letter a-H, 
        r being the previous row and R the new as a number 1-8.
        """
        n = self.board
        self.board = square_number(move[0])

           
def square_number(x):
    c = x[0]
    r = x[1]
    if c.isalpha() and len(c) == 1 and 'a' <= c <= 'h':
        C = ord(c) - ord('a')
    else:
        raise ValueError("Input must be a single letter between 'a' and 'h'")
     
    return C*8 + r

def start_game():
    game = dict()
    