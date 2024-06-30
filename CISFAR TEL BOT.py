from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

TOKEN = '6758796191:AAGMavCuB9ZIM1jlRiky9mjxXpHa_eJ1hwk'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Welcome!")

def help_command(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="""
    /start - Starts conversation
    /help - Shows this message 
    /train - Trains neural network
    """)

def train(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Model is being trained...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train.flatten(), epochs=5, batch_size=32, validation_data=(x_test, y_test.flatten()))
    model.save('cifar_classifier.model')
    context.bot.send_message(chat_id=update.effective_chat.id, text="Done! You can now send a photo!")

def handle_message(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Please train the model and send a picture!")

def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    prediction = model.predict(np.array([img / 255]))
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"In this image I see a {class_names[np.argmax(prediction)]}")

def load_model(update, context):
    try:
        model = tf.keras.models.load_model('cifar_classifier.model')
        return model
    except:
        return None  # Return an indicator that the model was not found

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("train", train))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    model = load_model(update=None, context=None)
    if model is None:
        print("Model not found. Please train the model.")
    else:
        print("Model loaded successfully.")

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
