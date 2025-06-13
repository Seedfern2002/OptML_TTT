class TicTacToe:
    def __init__(self):
        """Initialize the TicTacToe board and current player."""
        self.board = [" "] * 9
        self.current_player = "X"

    def available_moves(self):
        """Return a list of available move indices."""
        return [i for i, v in enumerate(self.board) if v == " "]

    def make_move(self, index):
        """Make a move at the given index if possible."""
        if self.board[index] == " ":
            self.board[index] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

    def winner(self):
        """Determine the winner of the game or if it's a draw."""
        b = self.board
        for i in range(3):
            if b[3*i] == b[3*i+1] == b[3*i+2] != " ":
                return b[3*i]
            if b[i] == b[i+3] == b[i+6] != " ":
                return b[i]
        if b[0] == b[4] == b[8] != " " or b[2] == b[4] == b[6] != " ":
            return b[4]
        if " " not in b:
            return "Draw"
        return None

    def clone(self):
        """Return a copy of the current game state."""
        clone = TicTacToe()
        clone.board = self.board[:]
        clone.current_player = self.current_player
        return clone
