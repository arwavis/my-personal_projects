from flask import Flask, render_template, request, redirect, url_for
from scripts.alarm_clock import set_alarm
from scripts.folder_delete import delete_all_files

app = Flask(__name__)


@app.route("/")
def home():
    """Main page with navigation links."""
    return render_template("index.html")


@app.route("/alarm", methods=["GET", "POST"])
def alarm():
    """Alarm Clock Page"""
    message = ""
    if request.method == "POST":
        alarm_time = request.form.get("alarm_time")
        message = set_alarm(alarm_time)

    return render_template("alarm.html", message=message)


@app.route("/delete", methods=["GET", "POST"])
def delete():
    """File Deletion Page"""
    message = ""
    if request.method == "POST":
        message = delete_all_files()

    return render_template("delete.html", message=message)


if __name__ == "__main__":
    app.run(debug=True)
