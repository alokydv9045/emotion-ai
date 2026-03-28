from collections import deque, Counter


class EmotionSmoother:

    def __init__(self, size=8):
        self.buffer = deque(maxlen=size)

    def update(self, emotion):
        self.buffer.append(emotion)
        return Counter(self.buffer).most_common(1)[0][0]
