import pyttsx3
import speech_recognition as sr


class Audio:

    def __init__(self):
        pass

    def text_to_speech(self, text: str):
        """ Функция для преобразования текста в голос """

        engine = pyttsx3.init()
        engine.setProperty('rate', 180)  # Настройка скорости речи
        engine.setProperty('volume', 1.0)  # Настройка громкости
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # Установить голос
        engine.say(text)
        engine.runAndWait()

    def speech_to_text(self) -> str:
        """ Функция для преобразования голоса в текст """

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Скажите что-нибудь...")
            recognizer.adjust_for_ambient_noise(source)  # Удаление фонового шума
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="ru-RU")  # Язык: "en-US" для английского
            return text
        except sr.UnknownValueError:
            print("Речь не распознана. Попробуйте еще раз.")
        except sr.RequestError as e:
            print(f"Ошибка сервиса распознавания: {e}")
        return ""


if __name__ == '__main__':

    audio = Audio()

    print("Проверка функции преобразования текста в голос:")
    sample_text = "Привет! Это проверка функции text_to_speech."
    print(f"Текст для воспроизведения: {sample_text}")
    audio.text_to_speech(sample_text)

    print("\nПроверка функции преобразования голоса в текст:")
    print("Скажите что-нибудь в микрофон.")
    recognized_text = audio.speech_to_text()
    if recognized_text:
        print(f"Распознанный текст: {recognized_text}")
    else:
        print("Текст не был распознан.")
