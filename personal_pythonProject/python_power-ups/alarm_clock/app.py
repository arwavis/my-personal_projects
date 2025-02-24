from flask import Flask, render_template, request, redirect, url_for
import threading
import datetime
import time
import simpleaudio as sa

app = Flask(__name__)

alarm_time = None
alarm_message = ""


def play_alarm_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file('/Users/aravindv/Documents/code/github/my-personal_projects'
                                                '/alarm_clock.wav')
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound: {e}")


def alarm_checker():
    global alarm_time, alarm_message
    while True:
        if alarm_time:
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M")
            if current_time == alarm_time:
                alarm_message = "Wake up!"
                play_alarm_sound()
                alarm_time = None  # Reset alarm after triggering
        time.sleep(30)


@app.route("/", methods=["GET", "POST"])
def index():
    global alarm_time, alarm_message
    if request.method == "POST":
        alarm_time = request.form.get("alarm_time")
        alarm_message = f"Alarm set for {alarm_time}"
        return redirect(url_for("index"))

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    return render_template("index.html", current_time=current_time, alarm_time=alarm_time, message=alarm_message)


if __name__ == "__main__":
    # Start alarm checker in a separate thread
    threading.Thread(target=alarm_checker, daemon=True).start()
    app.run(debug=True)
