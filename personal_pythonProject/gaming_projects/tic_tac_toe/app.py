from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace 'your_secret_key' with a real secret key


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['player_x'] = request.form['player_x']
        session['player_o'] = request.form['player_o']
        return redirect(url_for('game'))
    return render_template('index.html')


@app.route('/game', methods=['GET', 'POST'])
def game():
    winner = None
    if 'board' not in session:
        session['board'] = [' '] * 9
        session['turn'] = 'X'
        session['score'] = {'X': 0, 'O': 0}

    if request.method == 'POST':
        if 'reset_scores' in request.form:
            session['score'] = {'X': 0, 'O': 0}
            return redirect(url_for('game'))
        elif 'restart_game' in request.form:
            return redirect(url_for('index'))

        index = int(request.form['move'])
        if session['board'][index] == ' ':
            session['board'][index] = session['turn']
            if check_winner(session['board'], session['turn']):
                session['score'][session['turn']] += 1
                winner = session['turn']
                session['board'] = [' '] * 9  # Reset the board after a win
            session['turn'] = 'O' if session['turn'] == 'X' else 'X'

    return render_template('game.html', board=session['board'], turn=session['turn'], score=session['score'],
                           winner=winner, player_x=session.get('player_x', 'X'), player_o=session.get('player_o', 'O'))


def check_winner(board, player):
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]
    for a, b, c in lines:
        if board[a] == board[b] == board[c] == player:
            return True
    return False


if __name__ == '__main__':
    app.run(debug=True)
