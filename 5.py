import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import numpy as np

RUSSIAN_STOP_WORDS = {
    'в', 'с', 'на', 'по', 'о', 'от', 'за', 'из', 'к', 'у', 'для', 'до', 'без', 'под', 'над', 'при', 'про',
    'между', 'через',
    'и', 'а', 'но', 'или', 'либо', 'что', 'чтобы', 'как', 'когда', 'если', 'хотя', 'потому', 'так',
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'мой', 'твой', 'его', 'ее', 'наш', 'ваш', 'их',
    'себя', 'этот', 'тот', 'весь', 'сам',
    'не', 'ни', 'же', 'бы', 'ли', 'вот', 'уж', 'лишь', 'только', 'ведь', 'мол', 'дескать', 'даже',
    'это', 'то', 'та', 'те', 'так', 'тут', 'там', 'здесь', 'все', 'всё'
}


# парсинг новостей
def fetch_article_links(n=50):
    url = "https://lenta.ru"
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []

    news_elements = soup.select('a[href*="/news/"]')

    for element in news_elements:
        href = element.get('href', '')

        if href.startswith('/'):
            full_url = f"https://lenta.ru{href}"
        elif href.startswith('http'):
            full_url = href
        else:
            continue

        if full_url not in links:
            links.append(full_url)

        if len(links) >= n:
            break

    return links


def fetch_article(url):
    resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "Без заголовка"

    content_div = soup.select_one('.topic-body__content, .article__content')
    if content_div:
        paragraphs = content_div.find_all('p')
        body = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    else:
        return None

    return {"url": url, "title": title, "body": body}

print("Собираю новости...")
links = fetch_article_links(50)
articles = []

for i, url in enumerate(links[:50]):
    print(f"Обрабатываю статью {i + 1}/50...")
    art = fetch_article(url)
    if art and art['body']:
        articles.append(art)

df = pd.DataFrame(articles)
print(f"\n✅ Собрано новостей: {len(df)}")

print("\n" + "<>" * 50)
print("Все 50 статей:")
print("<>" * 50)
for i, row in df.iterrows():
    print(f"\n{i + 1:2}. ЗАГОЛОВОК: {row['title']}")
    print(f"   ССЫЛКА: {row['url']}")
    description = row['body'][:200] + "..." if len(row['body']) > 200 else row['body']
    print(f"   ОПИСАНИЕ: {description}")
    if (i + 1) % 5 == 0:
        print("-" * 100)


# предобработка
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['body_clean'] = df['body'].apply(preprocess)

# лемматизация
nlp = spacy.load("ru_core_news_sm", disable=['parser', 'ner'])

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [
        token.lemma_ for token in doc
        if (token.text.strip() and
            not token.is_punct and
            not token.is_space and
            len(token.text) > 2 and
            token.lemma_.lower() not in RUSSIAN_STOP_WORDS)
    ]
    return ' '.join(lemmas)


# Лемматизируем все тексты для частотного словаря
df['body_lemma_for_freq'] = df['body_clean'].apply(lemmatize_text)

# частотный словарь
print("\n" + "<>" * 50)
print("Создание частотного словаря...")
print("<>" * 50)

# Собираем все леммы из всех статей
all_lemmas = []
for text in df['body_lemma_for_freq']:
    if text:
        all_lemmas.extend(text.split())

# Создаем частотный словарь на основе лемм
freq_lemmas = Counter(all_lemmas)
most_common_lemmas = freq_lemmas.most_common(50)

lemma_in_articles = {}
for lemma in freq_lemmas:
    count = 0
    for text in df['body_lemma_for_freq']:
        if lemma in text.split():
            count += 1
    lemma_in_articles[lemma] = count

print("\n50 самых частых слов:")
print("<>" * 45)
print("№  | СЛОВО                 | КОЛИЧЕСТВО | ВСТРЕЧАЕТСЯ В СТАТЬЯХ")
print("<>" * 45)

for i, (lemma, count) in enumerate(most_common_lemmas, 1):
    articles_count = lemma_in_articles.get(lemma, 0)
    print(f"{i:2} | {lemma:20} | {count:10} | {articles_count:3} статей")

# Bag-of-Words
print("\n" + "<>" * 50)
print("СОЗДАНИЕ BAG-OF-WORDS...")
vectorizer = CountVectorizer(
    max_features=1000,
    stop_words=list(RUSSIAN_STOP_WORDS),
    min_df=2,
    token_pattern=r'\b\w{3,}\b'
)
X_bow = vectorizer.fit_transform(df['body_clean'])
df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())

print(f" Размерность Bag-of-Words: {df_bow.shape}")
print(" Топ-20 фич BoW:")
for i, feature in enumerate(vectorizer.get_feature_names_out()[:20], 1):
    print(f"   {i:2}. {feature}")

print("\n" + "<>" * 50)
print("ЛЕММАТИЗАЦИЯ ДЛЯ ЭМБЕДДИНГОВ...")

# Лемматизируем заново
df['body_lemma'] = df['body_clean'].apply(lemmatize_text)

# эмбеддинги с помощью spaCy
print("СОЗДАНИЕ ЭМБЕДДИНГОВ...")
embeddings = []
for doc in nlp.pipe(df['body_lemma'], batch_size=20):
    if len(doc) > 0:
        embeddings.append(doc.vector)
    else:
        embeddings.append(np.zeros(96))

df['embedding'] = embeddings

# Сохраняем результат
df.to_csv('news_processed.csv', index=False, encoding='utf-8')
df_bow.to_csv('news_bow.csv', index=False, encoding='utf-8')

# Сохраняем частотный словарь
with open('frequency_dict_lemmas.txt', 'w', encoding='utf-8') as f:
    f.write("ЧАСТОТНЫЙ СЛОВАРЬ:\n")
    f.write("<>" * 45 + "\n")
    f.write("№  | СЛОВО                 | КОЛИЧЕСТВО | В СТАТЬЯХ\n")
    f.write("<>" * 45 + "\n")
    for i, (lemma, count) in enumerate(most_common_lemmas, 1):
        articles_count = lemma_in_articles.get(lemma, 0)
        f.write(f"{i:2} | {lemma:20} | {count:10} | {articles_count:3}\n")

# Сохраняем список статей
with open('articles_list.txt', 'w', encoding='utf-8') as f:
    f.write("СПИСОК ВСЕХ СТАТЕЙ:\n")
    f.write("<>" * 50 + "\n")
    for i, row in df.iterrows():
        f.write(f"\n{i + 1:2}. ЗАГОЛОВОК: {row['title']}\n")
        f.write(f"   ССЫЛКА: {row['url']}\n")
        description = row['body'][:300] + "..." if len(row['body']) > 300 else row['body']
        f.write(f"   ОПИСАНИЕ: {description}\n")
        f.write("<>" * 50 + "\n")

# --- вывод результатов ---
print("\n" + "<>" * 50)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print("<>" * 50)
print(f" Частотный словарь: {len(freq_lemmas)} уникальных лемм")
print(f" Bag-of-Words: {df_bow.shape[0]} строк × {df_bow.shape[1]} фич")
print(f" Лемматизация: завершена для всех статей")
print(f" Эмбеддинги: созданы для всех статей ({len(df['embedding'].iloc[0])}-мерные векторы)")
print(f"\n Данные сохранены в файлах:")
print(f"   - news_processed.csv - все данные статей")
print(f"   - news_bow.csv - матрица Bag-of-Words")
print(f"   - frequency_dict_lemmas.txt - частотный словарь")
print(f"   - articles_list.txt - список всех статей с описаниями")
