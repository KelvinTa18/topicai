# Create your views here.
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, permission_required
from django.http import HttpResponse, JsonResponse, HttpRequest, HttpResponseBadRequest
from django.contrib import messages
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import os, fitz, asyncio, re, joblib, io, base64
from collections import Counter
from docx import Document
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from openai._exceptions import RateLimitError
from django.views.decorators.csrf import csrf_exempt
import json
import socket
import requests, sys, os, re, time, shutil
from django.http import StreamingHttpResponse

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlsplit
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def path_search(location:str):
    BASE_DIR = os.getcwd()
    return os.path.join(BASE_DIR, "templates", location)

def get_topic_from_article_byAPI(data, request):
    API_KEYS = [
        "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",
        "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",
        "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",
        "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",
        "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",
    ]
    BASE_URL = "https://openrouter.ai/api/v1"
    current_url = BASE_URL
    title_url = "Topicai"
    MODEL = "deepseek/deepseek-chat-v3:free"
    error = ""
    for key in API_KEYS:
        try:
            client = OpenAI(
                base_url=BASE_URL,
                api_key=key,
            )
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": current_url,
                    "X-Title": title_url,
                },
                extra_body={},
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": f"Tuliskan topik utama dari artikel berikut dalam 1–2 kalimat saja dan bahasa indonesia.\n\n{data}"
                    }
                ]
            )
            return completion.choices[0].message.content

        except RateLimitError as e:
            # Coba key berikutnya
            print(RateLimitError)
            error = f"❌ Semua API key telah mencapai batas pemakaian (rate limit)"
            continue

        except Exception as e:
            # Untuk error lainnya (jaringan, invalid API key, dsb.)
            error = f"❌ Sedang ada error, tolong hubungi administration."
            continue


    return error

# Create your views here.
def read(request:HttpRequest):
    if request.method == "POST":

        # Data Collection
        for key in request.POST:
            if key == "inputText":
                file = request.POST.getlist(key)
                if file[0].strip():
                    data = file
                    data = data[0].replace('\r\n', '\n').replace('\r', '\n').split('\n')


            else:
                file = request.FILES.get("inputFile")

                if file:
                    ext = os.path.splitext(file.name)[1].lower()
                    if ext == ".docx":
                        data = extract_text_from_docx(file)
                    elif ext == ".pdf":
                        data = extract_text_from_pdf(file)

        text = " ".join([word for word in data])
        print(type(text))
        # print("Raw text from PDF:", repr(text))  # DEBUG
        # print("Text length:", len(text))
        # Create word cloud


        # Tokenisasi (ubah kalimat jadi list kata)
        tokens = word_tokenize(text.lower())

        # Stopwords Bahasa Indonesia
        stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))

        # Filter kata-kata penting saja
        keywords = [word for word in tokens if word.isalpha() and word not in stop_words]
        freq_keywords = Counter(keywords)
        # 4. Urutkan dari yang paling sering
        freq_keywords = freq_keywords.most_common()
        # 5. Tampilkan hasil
        for word, count in freq_keywords:
            print(f"[{word}, {count}]")

        filtered_text = " ".join(keywords)
        # result_from_phi2 = get_topic_from_article(filtered_text)
        result_from_api = get_topic_from_article_byAPI(filtered_text, request)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(filtered_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

         # Simpan ke memory & encode base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        # Data Tokenize - Seperate Sentences
        data = fc_tokenize_seperate(data)

        # Data Preprocessing
        data = fc_translate_data(data)

        # Preprocessing - Cleaning Data
        print("Cleaning Data ...")
        data = [fc_cleaning_data(row) for row in data if fc_cleaning_data(row)]
        print("Cleaning Data - Done\n")

        # Preprocessing - Tokenize Data (Stop Word - English)
        data =  fc_tokenize_data(data)

        # Preprocessing - Cleaning Data (Delete Duplicate)
        print("Delete Duplicate ...")
        data = list(dict.fromkeys(data))
        print("Delete Duplicate - Done\n")



        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "model_and_vectorizer_main_data.pkl")
        bundle = joblib.load(model_path)
        model = bundle["model"]
        vectorizer_svd = bundle["vectorizer_tfdif"]
        vectorizer_tfidf = bundle["vectorizer"]

        # Transformasi input
        X_tfidf = vectorizer_tfidf.transform(data)         # <class 'scipy.sparse.csr.csr_matrix'>
        X_svd = vectorizer_svd.transform(X_tfidf)            # <class 'numpy.ndarray'>
        label = model.predict(X_svd)                         # hasil prediksi



        cluster_summary = dict(Counter(label))
        print(cluster_summary)
        for n, (d,l) in enumerate(zip(data, label)):
            print(f"{d} - {l}")

        # Threshold untuk deteksi out-of-domain
        threshold = 0.35
        passed_count = 0
        # Untuk setiap data, tampilkan label dan similarity
        for n, (text, vec, cluster_id) in enumerate(zip(data, X_svd, label)):
            # vec bentuknya (n_features,), jadi perlu reshape
            sim_scores = cosine_similarity([vec], model.cluster_centers_)
            max_sim = np.max(sim_scores)
            closest_cluster = np.argmax(sim_scores)

            # Cek apakah in-domain
            if max_sim >= threshold:
                passed_count += 1
                print(f"[{n}] ✔ In-domain | Cluster: {cluster_id:<2} | Sim: {max_sim:.2f} | Text: {text}")
            else:
                print(f"[{n}] ✖ Out-of-domain | Sim: {max_sim:.2f} | Text: {text}")

        # Persentase input yang lolos
        total_data = len(data)
        persentage = min(100,((passed_count / total_data) * 10000 / 35)) if total_data > 0 else 0
        persentage = {'passed' : round(persentage, 2), 'failed' : round(100 - persentage, 2)}
        print("\n=== Ringkasan ===")
        print(f"Total Data          : {total_data}")
        print(f"Lolos (In-domain)   : {passed_count}")
        print(f"Out-of-domain (OoD) : {total_data - passed_count}")
        print(f"Persentase Lolos    : {persentage}")
        # for n in data:
            # predicted_label =
        # model = model_data["model"]
        # vectorizer = model_data["vectorizer"]
        # X = vectorizer.transform(data)
        # print(X)

        clustering_information = {
            0 : 'Accounting & Economics',
            1 : 'Banking & Transactions',
            2 : 'Academic Publishing & Indexing',
            3 : 'Taxation',
            4 : 'E-Invoicing & Tax Payments',
            5 : 'Financial Reports & Data',
            6 : 'Corporate Identity & Tax',
            7 : 'Business Software (Mekari)',
            8 : 'Product & Management',
            9 : 'Payroll & Online Tax',
            10 : 'Mobile Banking Services',
            11 : 'Online Tax Services (OnlinePajak)',
            12 : 'General Journal Entries',
            13 : 'Taxation',
            14 : 'Personal Data & Privacy',
            15 : 'Accounting Statements',
            16 : 'Corporate Finance & Accounting',
            17 : 'Banking Security & Awareness',
            18 : 'Banking',
            19 : 'Real-Time Journal Data',
        }
        # clustering_information = {
        #     0 : 'accounting, economic, journal, company',
        #     1 : 'bank, transaction, credit, card',
        #     2 : 'sinta, journal, author, scopus',
        #     3 : 'tax',
        #     4 : 'payment, tax, journal, invoice',
        #     5 : 'journal, done, data, financial',
        #     6 : 'logo, bank, tax, journal',
        #     7 : 'journal, mekari, business, software',
        #     8 : 'product, journal, management, accounting',
        #     9 : 'pay, tax, journal, mekari',
        #     10 : 'bank, mobile, customers, number',
        #     11 : 'tax, onlinepajak, business, invoice',
        #     12 : 'journal, x, may, kec',
        #     13 : 'tax',
        #     14 : 'personal, data, bank, tax',
        #     15 : 'journal, accounting, statement, book',
        #     16 : 'business, journal, financial, accounting',
        #     17 : 'bank, asp, educatps, awasmodus',
        #     18 : 'bank',
        #     19 : 'time, journal, real, data',
        # }

        # Hitung total jumlah
        # total = sum(v for k, v in cluster_summary.items() if int(k) != 5)
        total = sum(cluster_summary.values())
        # Konversi jumlah ke persentase
        form = {
            clustering_information[int(k)]: (v / total * 100) if total > 0 else 0
            # for k, v in cluster_summary.items() if k !=5
            for k, v in cluster_summary.items()
        }

        # Urutkan dari persentase terbesar
        form = dict(sorted(form.items(), key=lambda x: x[1], reverse=True))

        # Format persentase ke 2 desimal (opsional)
        form = {k: round(v, 2) for k, v in form.items()}
        return render (request, path_search("main_page/read.html"), {
            'keywords' : freq_keywords[:10],
            'form': form,
            'image_base64': image_base64,
            'persentage' : persentage,
            'result_from_api_model_ai':result_from_api}
            )
    else:
        print("No post")

        return render (request, path_search("main_page/read.html"))
def extract_text_from_docx(file_obj):

    doc = Document(file_obj)
    return [para.text for para in doc.paragraphs if para.text.strip()]

def extract_text_from_pdf(file_obj):
    pdf = fitz.open(stream=file_obj.read(), filetype="pdf")
    all_text = []
    for page in pdf:
        all_text.append(
            ' '.join(line.strip() for line in page.get_text().split('\n') if line.strip())
        )
    return all_text

def fc_translate_data( data):
    print("Translate Data ...")
    # Inisialisasi translator untuk menerjemahkan judul penelitian
    translator = Translator()

    # Fungsi async untuk menerjemahkan seluruh list
    async def translate_all(data):
        tasks = [translator.translate(text, src='id', dest='en') for text in data]
        results = await asyncio.gather(*tasks)
        return [res.text for res in results]

    result = asyncio.run(translate_all(data))


    print("Translate Data - Done\n")
    return result

def fc_tokenize_seperate(data):
    result = []
    for paragraph in data:
        # Pertama: pisahkan berdasarkan kalimat (titik, dll)
        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
            # Kedua: pecah lagi kalimat berdasarkan titik dua
            sub_parts = sentence.split(':')
            for i, part in enumerate(sub_parts):
                part = part.strip()
                if part:
                    # Tambahkan titik dua kembali kecuali bagian terakhir
                    if i < len(sub_parts) - 1:
                        result.append(part + ':')
                    else:
                        result.append(part)
    return result

def fc_cleaning_data(data):
    # Cleaning symbol yang memengaruhi data ( - _ / )
    data = data.replace("-", " ")
    data = data.replace("_", " ")
    data = data.replace("/", " ")

    # Hapus UUID dalam kurung kurawal ({xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx})
    # data = re.sub(r'\{[0-9a-fA-F\-]{36}\}', '', data)
    data = re.sub(r'\{[^}]+\}', '', data)
    # Lower (kecilkan) semua huruf
    data = data.lower()
    # Hapus semua non-huruf kecil & non-spasi
    data = re.sub(r'[^a-z\s]', '', data)
    if data.strip():
        return data

def fc_tokenize_data(data):
    print("Tokenize Data (Stop Word - English) ...")
    for i in range(len(data)):
        tokens = data[i].split()
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        data[i] = ' '.join(tokens)


    print("Tokenize Data (Stop Word - English) - Done\n")
    return data

def fc_vectorize_data(data):
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.9,
        min_df=3,
        ngram_range=(1, 2),
        max_features=2000
    )
    return tfidf_vectorizer.fit_transform(data)

# 2. Fungsi untuk cek input user
def is_in_domain(user_input, vectorizer, cluster_centers, threshold=0.3):
    input_vec = vectorizer.transform([user_input])
    sim = cosine_similarity(input_vec, cluster_centers)
    max_sim = np.max(sim)
    return max_sim >= threshold, max_sim

def scrape(request:HttpRequest):
    if request.method == "POST":

        # Data Collection
        for key in request.POST:
            if key == "inputText":
                file = request.POST.getlist(key)
                if file[0].strip():
                    data = file
                    data = data[0].replace('\r\n', '\n').replace('\r', '\n').split('\n')


            else:
                file = request.FILES.get("inputFile")

                if file:
                    ext = os.path.splitext(file.name)[1].lower()
                    if ext == ".docx":
                        data = extract_text_from_docx(file)
                    elif ext == ".pdf":
                        data = extract_text_from_pdf(file)

        text = " ".join([word for word in data])
        print(type(text))
        # print("Raw text from PDF:", repr(text))  # DEBUG
        # print("Text length:", len(text))
        # Create word cloud


        # Tokenisasi (ubah kalimat jadi list kata)
        tokens = word_tokenize(text.lower())

        # Stopwords Bahasa Indonesia
        stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))

        # Filter kata-kata penting saja
        keywords = [word for word in tokens if word.isalpha() and word not in stop_words]
        freq_keywords = Counter(keywords)
        # 4. Urutkan dari yang paling sering
        freq_keywords = freq_keywords.most_common()
        # 5. Tampilkan hasil
        for word, count in freq_keywords:
            print(f"[{word}, {count}]")

        filtered_text = " ".join(keywords)
        # result_from_phi2 = get_topic_from_article(filtered_text)
        result_from_api = get_topic_from_article_byAPI(filtered_text, request)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(filtered_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

         # Simpan ke memory & encode base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        # Data Tokenize - Seperate Sentences
        data = fc_tokenize_seperate(data)

        # Data Preprocessing
        data = fc_translate_data(data)

        # Preprocessing - Cleaning Data
        print("Cleaning Data ...")
        data = [fc_cleaning_data(row) for row in data if fc_cleaning_data(row)]
        print("Cleaning Data - Done\n")

        # Preprocessing - Tokenize Data (Stop Word - English)
        data =  fc_tokenize_data(data)

        # Preprocessing - Cleaning Data (Delete Duplicate)
        print("Delete Duplicate ...")
        data = list(dict.fromkeys(data))
        print("Delete Duplicate - Done\n")



        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "model_and_vectorizer_main_data.pkl")
        bundle = joblib.load(model_path)
        model = bundle["model"]
        vectorizer_svd = bundle["vectorizer_tfdif"]
        vectorizer_tfidf = bundle["vectorizer"]

        # Transformasi input
        X_tfidf = vectorizer_tfidf.transform(data)         # <class 'scipy.sparse.csr.csr_matrix'>
        X_svd = vectorizer_svd.transform(X_tfidf)            # <class 'numpy.ndarray'>
        label = model.predict(X_svd)                         # hasil prediksi



        cluster_summary = dict(Counter(label))
        print(cluster_summary)
        for n, (d,l) in enumerate(zip(data, label)):
            print(f"{d} - {l}")

        # Threshold untuk deteksi out-of-domain
        threshold = 0.35
        passed_count = 0
        # Untuk setiap data, tampilkan label dan similarity
        for n, (text, vec, cluster_id) in enumerate(zip(data, X_svd, label)):
            # vec bentuknya (n_features,), jadi perlu reshape
            sim_scores = cosine_similarity([vec], model.cluster_centers_)
            max_sim = np.max(sim_scores)
            closest_cluster = np.argmax(sim_scores)

            # Cek apakah in-domain
            if max_sim >= threshold:
                passed_count += 1
                print(f"[{n}] ✔ In-domain | Cluster: {cluster_id:<2} | Sim: {max_sim:.2f} | Text: {text}")
            else:
                print(f"[{n}] ✖ Out-of-domain | Sim: {max_sim:.2f} | Text: {text}")

        # Persentase input yang lolos
        total_data = len(data)
        persentage = min(100,(passed_count / total_data) * 10000 / 35) if total_data > 0 else 0
        persentage = {'passed' : round(persentage, 2), 'failed' : round(100 - persentage, 2)}
        print("\n=== Ringkasan ===")
        print(f"Total Data          : {total_data}")
        print(f"Lolos (In-domain)   : {passed_count}")
        print(f"Out-of-domain (OoD) : {total_data - passed_count}")
        print(f"Persentase Lolos    : {persentage}")
        # for n in data:
            # predicted_label =
        # model = model_data["model"]
        # vectorizer = model_data["vectorizer"]
        # X = vectorizer.transform(data)
        # print(X)

        clustering_information = {
            0 : 'Accounting & Economics',
            1 : 'Banking & Transactions',
            2 : 'Academic Publishing & Indexing',
            3 : 'Taxation',
            4 : 'E-Invoicing & Tax Payments',
            5 : 'Financial Reports & Data',
            6 : 'Corporate Identity & Tax',
            7 : 'Business Software (Mekari)',
            8 : 'Product & Management',
            9 : 'Payroll & Online Tax',
            10 : 'Mobile Banking Services',
            11 : 'Online Tax Services (OnlinePajak)',
            12 : 'General Journal Entries',
            13 : 'Taxation',
            14 : 'Personal Data & Privacy',
            15 : 'Accounting Statements',
            16 : 'Corporate Finance & Accounting',
            17 : 'Banking Security & Awareness',
            18 : 'Banking',
            19 : 'Real-Time Journal Data',
        }
        # clustering_information = {
        #     0 : 'accounting, economic, journal, company',
        #     1 : 'bank, transaction, credit, card',
        #     2 : 'sinta, journal, author, scopus',
        #     3 : 'tax',
        #     4 : 'payment, tax, journal, invoice',
        #     5 : 'journal, done, data, financial',
        #     6 : 'logo, bank, tax, journal',
        #     7 : 'journal, mekari, business, software',
        #     8 : 'product, journal, management, accounting',
        #     9 : 'pay, tax, journal, mekari',
        #     10 : 'bank, mobile, customers, number',
        #     11 : 'tax, onlinepajak, business, invoice',
        #     12 : 'journal, x, may, kec',
        #     13 : 'tax',
        #     14 : 'personal, data, bank, tax',
        #     15 : 'journal, accounting, statement, book',
        #     16 : 'business, journal, financial, accounting',
        #     17 : 'bank, asp, educatps, awasmodus',
        #     18 : 'bank',
        #     19 : 'time, journal, real, data',
        # }

        # Hitung total jumlah
        # total = sum(v for k, v in cluster_summary.items() if int(k) != 5)
        total = sum(cluster_summary.values())
        # Konversi jumlah ke persentase
        form = {
            clustering_information[int(k)]: (v / total * 100) if total > 0 else 0
            # for k, v in cluster_summary.items() if k !=5
            for k, v in cluster_summary.items()
        }

        # Urutkan dari persentase terbesar
        form = dict(sorted(form.items(), key=lambda x: x[1], reverse=True))

        # Format persentase ke 2 desimal (opsional)
        form = {k: round(v, 2) for k, v in form.items()}
        return render (request, path_search("main_page/read_scrape.html"), {
            'keywords' : freq_keywords[:10],
            'form': form,
            'image_base64': image_base64,
            'persentage' : persentage,
            'result_from_api_model_ai':result_from_api}
            )
    else:
        print("No post")

        return render (request, path_search("main_page/read_scrape.html"))

@csrf_exempt
def check_domain(request):
    if request.method == "POST":
        data = json.loads(request.body)
        url = data.get("link", "")
        print("Jalan")
        try:
            print("1")
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            print("2")
            socket.gethostbyname(domain)  # akan error kalau domain tidak valid\
            print("3")
            result_crawing = crawing(url)
            print("4")
            print(f"result_crawing : {result_crawing}" )
            if result_crawing:
                return JsonResponse({
                    "found": True,
                    "crawlable": True,
                    "count": len(result_crawing)
                })
            else:
                return JsonResponse({
                    "found": True,
                    "crawlable": False,
                    "message": "Domain ditemukan, tapi tidak bisa di-crawling."
                })

        except Exception:
            return JsonResponse({
                "found": False,
                "crawlable": False,
                "message": "Domain tidak ditemukan / tidak valid."
            })
    return JsonResponse({"error": "Invalid request method"})

def normalize_url(input_url):
    parsed = urlparse(input_url)
    if not parsed.scheme:
        return "http://" + input_url  # default gunakan http
    return input_url


def stream_process(request):
    domain = request.GET.get("link", "")
    print(f"📥 Domain: {domain}")

    def generator():

        if not domain:
            yield "event: progress\ndata: ❌ (1/4) URL tidak diberikan\n\n"
            return
        yield f"event: progress\ndata: ✅ (1/4) Domain berhasil diterima: {domain}\n\n"

        time.sleep(1)
        domain_https = normalize_url(domain)  # <- auto tambahkan http jika tidak ada
        #Searching Domain
        try:
            yield f"event: progress\ndata: ⏳ (2/4) Data Sedang Di validasi di website ...\n\n"
            time.sleep(1)
            parsed = domain_https.replace("https://", "").replace("http://", "").split("/")[0]
            socket.gethostbyname(parsed)  # akan error kalau domain_https tidak valid
            yield f"event: progress\ndata: ✅ (2/4) Domain telah ditemukan di website : {parsed}\n\n"
        except Exception as e:
            yield f"event: progress\ndata: ❌ (2/4) Domain Tidak ditemukan\n\n"
            return  # stop streaming
        time.sleep(1)

        # Crawing
        try:
            yield f"event: progress\ndata: ⏳ (3/4) Data Sedang Di Crawing...\n\n"
            time.sleep(1)
            result_crawing = crawing(domain_https)
            print(f"data : len : {len(result_crawing)}\n\n")
            if len(result_crawing) < 1 :
                yield f"event: progress\ndata: ❌ (3/4) Tidak ada path pada domain tersebut\n\n"
                return
            yield f"event: progress\ndata: ✅ (3/4) Data Berhasil di crawing, total path : {len(result_crawing)} Path Link\n\n"
        except Exception as e:
            yield f"event: progress\ndata: ❌ (3/4) Domain Gagal Crawing\n\n"
            return  # stop streaming
        time.sleep(1)

        # Scraping
        try:
            testPrint()
            yield f"event: progress\ndata: ⏳ (4/4) Data Sedang Di Scraping...\n\n"
            time.sleep(1)
            scraping_result_data = {
                "texts": [],
                "images": [],
                "count": 0
            }
            time.sleep(1)
            yield from scraping(result_crawing, scraping_result_data)
            total_text = len(scraping_result_data["texts"])
            total_image = len(scraping_result_data["images"])
            yield f"event: progress\ndata: ✅ (4/4) Total Text: {total_text}, Total Image: {total_image}\n\n"
            time.sleep(1)
            # if len(result_scraping) < 1 :
                # yield f"Tidak ada data yang di Scraping\n\n"
                # return
            yield f"event: progress\ndata: ✅ (4/4) Data Berhasil di Scraping {scraping_result_data['count']}\n\n"
            yield f"event: result_text\ndata: {json.dumps(format_scraped_text_and_image(scraping_result_data))}\n\n"
        except Exception as e:
            print(e)
            yield f"event: progress\ndata: ❌ (4/4) Domain Gagal Scraping\n\n"
            return  # stop streaming
    response = StreamingHttpResponse(generator(), content_type="text/event-stream")
    return response

def crawing(domain):
    all_results_path = []
    print("\nPROCESS CRAWING..........")


    print(f"\n🌐 Domain: {domain}")
    result_path_from_domain = get_paths_from_domain(domain)
    if result_path_from_domain:
        print(f"✅ Path yang ada dari Domain '{domain}':")
        for i, url in enumerate(result_path_from_domain, start=1):
            print(f"{i}. {url}")
            all_results_path.append(url)
    else:
        return None
        return(f"⚠️ Tidak Ditemukan path yang relavan dari domain '{domain}'s")

    all_results_path = delete_duplicate_list(all_results_path)
    return all_results_path
    # with open(self.result_domain_path, "w", encoding="utf-8") as f:
    #     for url in all_results_path:
    #         f.write(url + "\n")
    # print(f"\n💾 Data path dari domain {self.domain_url} disimpan sebanyak {len(all_results_path)} path pada file {self.result_domain_path}")

def get_paths_from_domain(domain):
    print("🔍 Mengambil link dari homepage...")
    try:
        response = requests.get(domain, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"❌ Domain '{domain}' Gagal mengambil halaman: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    domain_parse = urlparse(domain).netloc
    links = set()
    total = 0

    for link in soup.find_all("a", href=True):
        href = urljoin(domain, link['href'])
        if urlparse(href).netloc == domain_parse and not is_duplicate_path(href, links):
            links.add(str(href))
            total += 1
            sys.stdout.write(f"\r🔗 Mengumpulkan link internal... {total} ditemukan")
            sys.stdout.flush()

    sys.stdout.write("\r" + " " * 50 + "\r")  # Clean The Row
    sys.stdout.flush()
    return sorted(links)

def delete_duplicate_list(list_data):
    seen = set()
    result = []
    dupes = []
    for item in list_data:
        if item not in seen:
            seen.add(item)
            result.append(item)
        else:
            dupes.append(item)
    if dupes:
        print(f"\n🗑️  Delete the duplicate data ({len(dupes)} data)")
    return result

def is_duplicate_path(new_url, existing_urls):
    new_parts = urlsplit(new_url)
    for existing in existing_urls:
        existing_parts = urlsplit(existing)
        if (
            existing_parts.scheme == new_parts.scheme and
            existing_parts.netloc == new_parts.netloc and
            existing_parts.path == new_parts.path
        ):
            # If the existing one has a query (?), ignore the new one.
            if existing_parts.query:
                return True
            # But if the existing one **doesn't** have a query, and the new one has a query => replace
            elif new_parts.query:
                existing_urls.remove(existing)
                return False
            else:
                return True
    return False

def testPrint():
    print("test")

def scraping(result_domain_path, scraping_result_data):
    print("\nPROCESS SCRAPING..........")

    # Setup Selenium
    options = Options()
    options.add_argument("--headless=new")  # FIX for Chrome newest version
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--enable-unsafe-webgpu")
    options.add_argument("--enable-unsafe-swiftshader")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument("--ignore-ssl-errors=yes")
    options.add_argument("--disable-features=BlockInsecurePrivateNetworkRequests")
    driver = webdriver.Chrome(options=options)
    success_scraping = 0
    success_scarping_text = []
    success_scarping_image = []
    for i, domain in enumerate(result_domain_path, start=1):
        print(f"\n🔍 Scraping ({i}/{len(result_domain_path)}) : {domain}")
        # filename_base = rename_path_from_result_path(domain, i)
        # folder_path = os.path.join(result_domain_folder, filename_base)
        # os.makedirs(folder_path, exist_ok=True)
        yield f"data : ✅ (4/4) Domain berhasil discraping ({i}/{len(result_domain_path)}): {domain}\n\n"

        # Check Scraping
        try:
            driver.get(domain)
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            text = scrape_text(domain, soup)
            image = scrape_image(domain, soup)
            scraping_result_data['count'] += 1

        except Exception as e:
            print(f"❌ Gagal Scraping {domain}: {e}")
            text = False
            image = False
        # Check Text from Domain
        if text:
            # file_path_text = os.path.join(folder_path, f"text_{filename_base}.txt")
            # with open(file_path_text, "w", encoding="utf-8") as f:
                # f.write(text)
            # print(f"✅ Disimpan : {file_path_text}")
            scraping_result_data['texts'].append(text)
            yield f"event: progress\ndata: ✅ (4/4) Domain berhasil discraping ({i}/{len(result_domain_path)}): {domain}\n\n"
        else:
            yield f"event: progress\ndata: ⚠️ (4/4) Tidak ada text dari {domain}\n\n"

        # Check Image from Domain
        if image:
            # file_path_image = os.path.join(folder_path, f"image_{filename_base}.txt")
            # with open(file_path_image, "w", encoding="utf-8") as f:
                # for img in image:
                    # f.write(f"{img}\n")
            # print(f"✅ Disimpan : {file_path_image}")
            scraping_result_data['images'].append(image)
            yield f"event: progress\ndata: ✅ (4/4) Domain berhasil discraping ({i}/{len(result_domain_path)}): {domain}\n\n"
        else:
            yield f"event: progress\ndata: ⚠️ (4/4) Tidak ada image dari {domain}\n\n"

    driver.quit()
    # print(f"\n💾 Data path dari domain {self.domain_url} sebanyak {len(domains)}\nSuccess Scraping Domain\t: {success_scraping} Domain\nText Scraping\t\t: {success_scarping_text} Data\nImage Scraping\t\t: {success_scarping_image} Data")
    # print(f"\n🎉 Selesai! Semua file disimpan di folder: {self.result_domain_folder}\n")


def scrape_text(self, domain, soup):
    try:
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()
        # Add newlines after important elements to make them more readable.
        block_elements = ['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br']

        for tag in soup.find_all(block_elements):
            tag.append('\n')

        text = soup.get_text()
        raw_lines = text.splitlines()
        paragraphs = []
        current = []

        empty_count = 0

        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                empty_count += 1
                if empty_count >= 2 and current:
                    paragraphs.append(' '.join(current))
                    current = []
            else:
                current.append(stripped)
                empty_count = 0

        if current:  # kalau ada sisa blok terakhir
            paragraphs.append(' '.join(current))

        return '\n'.join(paragraphs)

    except Exception as e:
        print(f"❌ Gagal process {domain}: {e}")
        return False

def scrape_text(domain, soup):
    try:
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()
        # Add newlines after important elements to make them more readable.
        block_elements = ['p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br']

        for tag in soup.find_all(block_elements):
            tag.append('\n')

        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

    except Exception as e:
        print(f"❌ Gagal process {domain}: {e}")
        return False



def scrape_image(domain, soup):
    try:
        image_data = []
        valid_exts = {".jpg", ".jpeg", ".png", ".svg", ".gif", ".webp"}
        exclude_names = {"adsct", "generic"}
        for img in soup.find_all('img'):
            img_src = img.get('src')
            if img_src:
                full_url = urljoin(domain, img_src)
                img_name = full_url.split('/')[-1].split('?')[0]

                # Get file name & extension
                name_only, ext = os.path.splitext(img_name)
                name_only = name_only.lower()
                ext = ext.lower()

                # Filter by extension and name
                if ext in valid_exts and name_only not in exclude_names:
                    image_data.append(name_only)
        return image_data

    except Exception as e:
        print(f"❌ Gagal process {domain}: {e}")
        return False

def rename_path_from_result_path(self, url, index):
    # parsed = urlparse(url)
    # domain_parsed = parsed.netloc
    # path = parsed.path.strip("/").replace("/", "-")
    # filename = domain_parsed if not path else f"{domain_parsed}-{path}"
    # filename = re.sub(r'[^\w\-]', '_', filename)
    # return f"{filename}_{index:03d}"
    domain = urlparse(url).netloc
    domain_clean = re.sub(r'[^\w\-]', '_', domain)
    return f"{domain_clean}_{index:03d}"

def format_scraped_text_and_image(scraping_result_data):
    texts = scraping_result_data.get("texts", [])
    images = scraping_result_data.get("images", [])

    combined_output = []
    for i, text in enumerate(texts):
        entry = f"{text.strip()}"  # teks
        if i < len(images):
            img_list = images[i]
            img_text = "\n".join(f"- {img}" for img in img_list)
            entry += img_text
        combined_output.append(entry)

    return "\n\n---\n\n".join(combined_output)  # separator antar halaman
