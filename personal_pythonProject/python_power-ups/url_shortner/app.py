from flask import Flask, request, redirect, render_template
import hashlib
import string
import random

app = Flask(__name__)

# In-memory store for URL mappings
url_map = {}


# Function to generate a short unique ID
def generate_short_id(num_chars=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=num_chars))


# Function to hash the long URL
def hash_url(long_url):
    return hashlib.sha256(long_url.encode()).hexdigest()[:10]


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        name = request.form['name']
        long_url = request.form['long_url']

        # Check if the long URL is already in the map
        short_id = hash_url(long_url)
        if short_id not in url_map:
            # If not, add it to the map
            short_id = generate_short_id()
            while short_id in url_map:
                short_id = generate_short_id()
            url_map[short_id] = long_url

        # Generate the short URL
        short_url = request.host_url + short_id

        # Save the name, long URL, and short URL to a notepad file
        with open('url_data.txt', 'a') as f:
            f.write(f"{name}, {long_url}, {short_url}\n")

        return render_template('index.html', short_url=short_url, long_url=long_url, name=name)

    return render_template('index.html')


@app.route('/<short_id>')
def redirect_to_long_url(short_id):
    long_url = url_map.get(short_id)
    if long_url:
        return redirect(long_url)
    return "URL not found", 404


if __name__ == '__main__':
    app.run(debug=True)
