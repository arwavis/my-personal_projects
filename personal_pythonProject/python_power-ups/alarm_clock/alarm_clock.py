import datetime
import time
import simpleaudio as sa


def set_alarm(alarm_time):
    """The program checks the time every 30 seconds until the current time matches the alarm time, at which point it
    plays the sound and prints the message."""
    print(f"Alarm set for {alarm_time}")
    while True:
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M")
        if current_time == alarm_time:
            print("Wake up!")
            try:
                wave_obj = sa.WaveObject.from_wave_file('/Users/aravindv/Documents/code/github/my-personal_projects'
                                                        '/alarm_clock.wav')
                play_obj = wave_obj.play()
                play_obj.wait_done()  # Wait for the sound to finish playing
            except Exception as e:
                print(f"Error playing sound: {e}")
            break
        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    alarm_time = input("Enter the time for the alarm (HH:MM): ")
    set_alarm(alarm_time)
