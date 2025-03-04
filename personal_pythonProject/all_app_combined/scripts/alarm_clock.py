import threading
import datetime
import time
import simpleaudio as sa

alarm_time = None
alarm_message = ""


def play_alarm_sound():
    try:
        wave_obj = sa.WaveObject.from_wave_file(
            '/Users/aravindv/Documents/code/github/my-personal_projects/alarm_clock.wav')
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound: {e}")


def alarm_checker():
    global alarm_time, alarm_message
    while True:
        if alarm_time:
            now = datetime.datetime.now().strftime("%H:%M")
            if now == alarm_time:
                alarm_message = "Wake up!"
                play_alarm_sound()
                alarm_time = None  # Reset alarm after triggering
        time.sleep(30)


def set_alarm(time_str):
    """Set an alarm for the given time (HH:MM format)."""
    global alarm_time, alarm_message
    alarm_time = time_str
    alarm_message = f"Alarm set for {alarm_time}"
    return alarm_message


# Start the alarm checker thread
threading.Thread(target=alarm_checker, daemon=True).start()
