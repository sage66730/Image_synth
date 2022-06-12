import os

if __name__ == "__main__":
    subjects = os.listdir("raw_data")

    for subject in subjects:
        sentences = os.listdir(f"raw_data/{subject}")
        if sorted(sentences)[0] == "00": continue

        sentences = sorted(sentences, key= lambda x: int(x.split("_")[-1]))
        for n, sentence in enumerate(sentences):
            os.rename(f"raw_data/{subject}/{sentence}",f"raw_data/{subject}/{n:02d}")

        