import json
from gtts import gTTS

data = {
    "id": 995762,
    "image": "DGM4/manipulation/infoswap/995762-043201-infoswap.jpg",
    "text": "A Victorian court decided on Friday that former AFL player agent Ricky Nixon can publish his tellall book despite Kim Duthie s push to ban it",
    "fake_cls": "face_swap",
    "fake_image_box": [
        178,
        35,
        226,
        99
    ],
    "fake_text_pos": [],
    "mtcnn_boxes": [
        [
            178,
            35,
            226,
            99
        ],
        [
            357,
            39,
            401,
            95
        ],
        [
            58,
            107,
            79,
            137
        ]
    ]
}


text_to_speak = data['text']

tts = gTTS(text_to_speak, lang='en')
tts.save("output_audio.mp3")

print("Audio file has been saved as 'output_audio.mp3'")
