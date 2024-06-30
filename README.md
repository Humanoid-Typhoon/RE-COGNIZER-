from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram import Update
from io import BytesIO
import cv2
import numpy as np
import tensorflow as tf

TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

def start(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Welcome!")

def help_command(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text="""
    /start - Starts conversation
    /help - Shows this message
    /train - Trains neural network
    /load - Loads the pre-trained model
    """)

def train(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Model is being trained...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train.flatten(), epochs=5, batch_size=32, validation_data=(x_test, y_test.flatten()))
    model.save('cifar_classifier.model')
    context.bot.send_message(chat_id=update.effective_chat.id, text="Done! You can now send a photo!")

def handle_message(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Please train the model and send a picture!")

def handle_photo(update: Update, context: CallbackContext):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    prediction = model.predict(np.array([img / 255]))
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"In this image I see a {class_names[np.argmax(prediction)]}")

def load_model(update: Update, context: CallbackContext):
    global model
    try:
        model = tf.keras.models.load_model('cifar_classifier.model')
        context.bot.send_message(chat_id=update.effective_chat.id, text="Model loaded successfully.")
    except:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Model not found. Please train the model first.")

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("train", train))
    dp.add_handler(CommandHandler("load", load_model))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
