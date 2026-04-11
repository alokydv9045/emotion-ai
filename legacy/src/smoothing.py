from collections import deque, Counter


class EmotionSmoother:

    def __init__(self, size=8):
        self.size = size
        self.buffer = deque(maxlen=size)

    def update(self, emotion):
        self.buffer.append(emotion)
        if not self.buffer:
            return emotion
        return Counter(self.buffer).most_common(1)[0][0]
