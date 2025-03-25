import sys

import pyttsx3

import pytchat

from utils.yt_chat import yt_chat


def main():

    engine = pyttsx3.init()

    voices = engine.getProperty('voices')
    
    def set_voice(gender='female'):
        if gender.lower() == 'female':
            engine.setProperty('voice', voices[1].id)  # Female voice
        elif gender.lower() == 'male':
            engine.setProperty('voice', voices[0].id)  # Male voice
        else:
            print("Voice gender not recognized. Using default voice.")

    set_voice('female')
    
    while True:
        try:
            video_id = input("Enter YouTube Live Stream video ID: ")
            chat = yt_chat(video_id)
            break
        except pytchat.exceptions.InvalidVideoIdException:
            print(
                "Invalid video id. Please make sure you've entered correct video id.\n"
            )

    while chat.is_alive():
        for c in chat.get().sync_items():
            try:
                print(f"{c.datetime} [{c.author.name}]- {c.message}")
                engine.say(f"{c.author.name} says {c.message}")
                engine.runAndWait()
            except KeyboardInterrupt:
                sys.exit("\n")


if __name__ == "__main__":
    main()
