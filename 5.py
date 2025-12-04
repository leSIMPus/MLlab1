import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
import time
import numpy as np


# 1. Функция парсинга новостей с Lenta.ru
def parse_lenta_ru(num_news=50):
    base_url = "https://lenta.ru"
    news_list = []
    page_num = 0

    while len(news_list) < num_news:
        if page_num == 0:
            url = base_url
        else:
            url = f"{base_url}/page/{page_num}/"

        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Находим все новостные элементы
            news_items = soup.find_all('a', href=True)

            for item in news_items:
                if len(news_list) >= num_news:
                    break

                try:
                    # Проверяем, что это новость (по ссылке)
                    href = item.get('href')
                    if not href or '/news/' not in href:
                        continue

                    # Получаем заголовок
                    title = item.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue

                    # Получаем полную ссылку
                    if href.startswith('/'):
                        link = base_url + href
                    else:
                        link = href

                    # Получаем дату
                    date_elem = item.find('time')
                    date = date_elem.get('datetime') if date_elem else ''

                    # Парсим полный текст новости
                    full_text = ""
                    try:
                        article_response = requests.get(link, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                        article_soup = BeautifulSoup(article_response.content, 'html.parser')

                        # Ищем основной текст
                        article_body = article_soup.find('div', class_=re.compile(r'(article|text|content)'))
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            for p in paragraphs:
                                text = p.get_text(strip=True)
                                if len(text) > 20:
                                    full_text += text + " "

                    except:
                        full_text = "Текст не доступен"

                    # Добавляем новость
                    news_list.append({
                        'title': title,
                        'url': link,
                        'date': date,
                        'text': full_text.strip(),
                        'full_content': f"{title}. {full_text}".strip()
                    })

                except:
                    continue

            page_num += 1
            time.sleep(1)

        except Exception as e:
            print(f"Ошибка: {e}")
            break

    return pd.DataFrame(news_list[:num_news])


# 2. Парсим новости
print("Парсим новости с Lenta.ru...")
df_news = parse_lenta_ru(50)
print(f"Получено {len(df_news)} новостей")

if len(df_news) > 0:
    print("\nПервые 50 новостей:")
    for i, row in df_news.head(50).iterrows():
        print(f"{i + 1}. {row['title'][:80]}...")
else:
    print("Не удалось получить новости!")
    exit()

# 3. Упрощенная предобработка текста
import string


def simple_preprocess(text):
    """Упрощенная очистка текста без NLTK"""
    if not isinstance(text, str):
        return ""

    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление знаков препинания
    text = text.translate(str.maketrans('', '', string.punctuation + '«»—–'))

    # Удаление цифр
    text = ''.join([char for char in text if not char.isdigit()])

    # Простая токенизация по пробелам
    tokens = text.split()

    # Простой список стоп-слов для русского
    russian_stopwords = {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
                         'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
                         'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
                         'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до',
                         'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
                         'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем',
                         'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет',
                         'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
                         'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были',
                         'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой',
                         'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них',
                         'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой',
                         'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда',
                         'конечно', 'всю', 'между'}

    # Удаление стоп-слов и коротких слов
    tokens = [word for word in tokens if word not in russian_stopwords and len(word) > 2]

    return ' '.join(tokens)


# Применяем предобработку
df_news['processed_text'] = df_news['full_content'].apply(simple_preprocess)
print("\nПример обработанного текста:")
sample_text = df_news['processed_text'].iloc[0][:300] if len(df_news) > 0 else ""
print(sample_text)


# 4. Частотный словарь
def create_frequency_dict(texts):
    all_words = []
    for text in texts:
        all_words.extend(text.split())

    freq_dict = {}
    for word in all_words:
        freq_dict[word] = freq_dict.get(word, 0) + 1

    return dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))


if len(df_news) > 0:
    freq_dict = create_frequency_dict(df_news['processed_text'].tolist())
    print(f"\nТоп-20 слов частотного словаря:")
    for i, (word, freq) in enumerate(list(freq_dict.items())[:20]):
        print(f"{i + 1}. {word}: {freq}")
else:
    print("Нет данных для создания частотного словаря")

# 5. Кодирование Bag of Words
from sklearn.feature_extraction.text import CountVectorizer

if len(df_news) > 0 and len(df_news['processed_text']) > 0:
    vectorizer = CountVectorizer(max_features=100)
    X_bow = vectorizer.fit_transform(df_news['processed_text'])
    print(f"\nBag of Words matrix shape: {X_bow.shape}")
    print(f"Словарь BoW (первые 10 слов): {vectorizer.get_feature_names_out()[:10]}")
else:
    print("Нет данных для создания BoW")

# 6. ДОПОЛНИТЕЛЬНО: Упрощенная лемматизация
print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНАЯ ЧАСТЬ:")
print("=" * 60)


def simple_lemmatize(text):
    """Упрощенная лемматизация (базовые правила)"""
    if not text:
        return ""

    # Базовые замены для русского языка
    replacements = {
        'ая': 'ый', 'ое': 'ый', 'ые': 'ый', 'ых': 'ый',
        'ому': 'ый', 'ими': 'ый', 'ую': 'ый'
    }

    tokens = text.split()
    lemmas = []

    for word in tokens:
        # Простая нормализация
        word = word.rstrip('ыиойуюяе')
        lemmas.append(word)

    return ' '.join(lemmas)


if len(df_news) > 0:
    df_news['lemmatized_text'] = df_news['processed_text'].apply(simple_lemmatize)
    print("Пример лемматизированного текста:")
    sample_lemma = df_news['lemmatized_text'].iloc[0][:300] if len(df_news) > 0 else ""
    print(sample_lemma)

# 7. ДОПОЛНИТЕЛЬНО: TF-IDF как альтернатива эмбеддингам
from sklearn.feature_extraction.text import TfidfVectorizer

if len(df_news) > 0 and len(df_news['lemmatized_text']) > 0:
    tfidf_vectorizer = TfidfVectorizer(max_features=50)
    X_tfidf = tfidf_vectorizer.fit_transform(df_news['lemmatized_text'])
    print(f"\nTF-IDF матрица: {X_tfidf.shape}")
    print(f"Словарь TF-IDF (первые 10 слов): {tfidf_vectorizer.get_feature_names_out()[:10]}")
else:
    print("Нет данных для создания TF-IDF")

# 8. Сохранение результатов
if len(df_news) > 0:
    df_news.to_csv('parsed_news.csv', index=False, encoding='utf-8')
    print(f"\nРезультаты сохранены в 'parsed_news.csv'")
    print(f"Количество новостей: {len(df_news)}")

    if 'X_bow' in locals():
        print(f"Размер BoW матрицы: {X_bow.shape}")

    if 'X_tfidf' in locals():
        print(f"Размер TF-IDF матрицы: {X_tfidf.shape}")
else:
    print("\nНет данных для сохранения")

# 9. Альтернативный вариант: если нужен NLTK
print("\n" + "=" * 60)
print("ДЛЯ ПОЛНОЙ ФУНКЦИОНАЛЬНОСТИ NLTK:")
print("=" * 60)
print("Запустите в консоли:")
print("""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
""")
print("Или используйте код без NLTK (как выше)")