from pymongo import MongoClient

INTENTS = [
    {
        "tag": "greetings",
        "patterns": [
            "Hello",
            "hi",
            "what's up?",
            "good morning",
            "how's it going?",
            "how are you?",
        ],
        "responses": [
            "Hello!",
            "Hey!",
            "Greetings! How can I assist you?",
            "Hi there!",
        ],
    },
    {
        "tag": "name",
        "patterns": [
            "What's your name?",
            "can you tell me your name?",
            "What do I call you?",
        ],
        "responses": [
            "I'm a Chatbot. My name is Jake.",
            "You can call me Chatbot.",
        ],
    },
    {
        "tag": "age",
        "patterns": ["What's your age?", "How old are you?", "age?"],
        "responses": ["My age is 25!", "I'm as old as the digital realm."],
    },
    {
        "tag": "goodbye",
        "patterns": [
            "Goodbye",
            "bye",
            "see you later",
            "bye-bye",
            "talk to you later",
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "Farewell! Feel free to return whenever you wish.",
        ],
    },
    {
        "tag": "weather",
        "patterns": [
            "What's the weather like?",
            "Tell me about the weather",
            "weather forecast",
        ],
        "responses": [
            "I'm not equipped with real-time data,"
            " but you can check a weather website for the latest updates."
        ],
    },
    {
        "tag": "help",
        "patterns": [
            "Can you help me?",
            "I need assistance",
            "help",
            "support",
        ],
        "responses": [
            "Of course, I'm here to help!" " What do you need assistance with?"
        ],
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "thanks", "thanks a lot", "I appreciate it"],
        "responses": [
            "You're welcome!",
            "No problem. If you have more questions, feel free to ask.",
        ],
    },
    {
        "tag": "jokes",
        "patterns": ["Tell me a joke", "joke", "funny", "make me laugh"],
        "responses": [
            "Sure, here's one: Why did the scarecrow win an award?"
            " Because he was outstanding in his field!"
        ],
    },
    {
        "tag": "unknown",
        "patterns": [
            "",
            "I don't know",
            "I'm not sure",
            "Sorry, I didn't understand",
            "Can you repeat that?",
        ],
        "responses": [
            "I'm still learning and might not understand everything."
            " Is there something else I can assist you with?",
            "I'm not sure how to respond to that.",
        ],
    },
]


with MongoClient("localhost", 27017) as client:
    jake_db = client.jake_database
    intents_collection = jake_db.intents
    intents_collection.insert_many(INTENTS)
