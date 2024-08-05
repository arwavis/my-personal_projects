from flask import Flask, request, render_template, url_for
import random

app = Flask(__name__)

# Generate a random number when the server starts
target_number = random.randint(1, 100)
attempts = 0
max_attempts = 10


@app.route('/', methods=['GET', 'POST'])
def guess():
    global target_number, attempts
    message = None
    if request.method == 'POST':
        guess = int(request.form['guess'])
        attempts += 1
        if guess < target_number:
            message = 'Too low!'
        elif guess > target_number:
            message = 'Too high!'
        else:
            message = 'Congratulations! You guessed it!'
            target_number = random.randint(1, 100)  # Reset the number
            attempts = 0  # Reset attempts
            return render_template('index.html', message=message, max_attempts=max_attempts - attempts)

        if attempts >= max_attempts:
            message = 'Game Over! The number was {}. Starting new game...'.format(target_number)
            target_number = random.randint(1, 100)  # Reset the number
            attempts = 0  # Reset attempts

    return render_template('index.html', message=message, max_attempts=max_attempts - attempts)


if __name__ == '__main__':
    app.run(debug=True)
