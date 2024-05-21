from flask import Flask, render_template, request, redirect, url_for
import random
from datetime import date

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def generate_password():
    password = None
    if request.method == 'POST':
        no_of_digits = int(request.form['digits'])
        no_of_lowcase = int(request.form['lowcase'])
        no_of_uppercase = int(request.form['uppercase'])
        no_of_symbols = int(request.form['symbols'])
        application_name = request.form['application']

        DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        LOCASE_CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                             'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q',
                             'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                             'z']
        UPCASE_CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'I', 'J', 'K', 'M', 'N', 'O', 'P', 'Q',
                             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                             'Z']
        SYMBOLS = ['@', '#', '$', '%', '=', ':', '?', '.', '/', '|', '~', '>',
                   '*', '(', ')', '<']

        password_list = []
        for digit in range(1, no_of_digits + 1):
            password_list.append(random.choice(DIGITS))

        for lower in range(1, no_of_lowcase + 1):
            password_list.append(random.choice(LOCASE_CHARACTERS))

        for upper in range(1, no_of_uppercase + 1):
            password_list.append(random.choice(UPCASE_CHARACTERS))

        for symbol in range(1, no_of_symbols + 1):
            password_list.append(random.choice(SYMBOLS))

        random.shuffle(password_list)
        password = ''.join(password_list)

        current_date_time = date.today()
        textual_month_format = current_date_time.strftime("%B %d, %Y")

        with open('/Users/aravindv/Documents/code/github/passwords.txt', 'a') as f:
            f.write(
                f"{application_name} password is : {password} and this password was generated on {textual_month_format}\n")

    return render_template('index.html', password=password)


if __name__ == '__main__':
    app.run(debug=True)
