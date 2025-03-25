import pytchat


def yt_chat(video_id):
    chat = pytchat.create(video_id=video_id)

    return chat


if __name__ == "__main__":
    video_id = input("Enter YouTube Live Stream video ID: ")
    chat = yt_chat(video_id)
    while chat.is_alive():
        for c in chat.get().sync_items():
            print(f"{c.datetime} [{c.author.name}]- {c.message}")
