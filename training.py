import re
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

FALSE_BIAS_STOPWORDS = [
    'inilah', 'itulah', 'itu', 'ini', 'padahal', 'memang', 'sebenarnya', 'sebenernya',
    'yang', 'yg', 'di', 'ke', 'dari', 'dr', 'dan', 'atau', 'untuk', 'utk', 'buat',
    'pada', 'dengan', 'dgn', 'nya', 'saya', 'sy', 'aku', 'gw', 'gue', 'kami', 'kita',
    'mereka', 'dia', 'kalian', 'anda', 'orang', 'org', 'rakyat', 'masyarakat', 'warga',
    'negara', 'indonesia', 'pemerintah', 'pemerintahan', 'presiden', 'prabowo', 'gibran',
    'menteri', 'pejabat', 'dpr', 'bgn', 'sppg', 'mbg', 'program', 'kebijakan',
    'bisa', 'dapat', 'akan', 'sudah', 'udah', 'sdh', 'udh', 'belum', 'blm', 'sedang',
    'masih', 'lagi', 'kalau', 'kalo', 'klo', 'kl', 'jika', 'bila', 'tapi', 'tetapi',
    'namun', 'karena', 'krn', 'sebab', 'jadi', 'juga', 'jg', 'saja', 'aja', 'cuma',
    'hanya', 'lebih', 'paling', 'sangat', 'banget', 'bgt', 'sekali', 'sama', 'sma',
    'banyak', 'semua', 'para', 'setiap', 'tiap', 'ada', 'punya', 'lihat', 'coba',
    'kata', 'bilang', 'anggap', 'pikir', 'rasa', 'tau', 'tahu', 'paham', 'erti',
    'eh', 'btw', 'hmm', 'oh', 'ngomong', 'ya', 'yaa', 'iya', 'dong', 'deh', 'sih', 'nih', 'kok', 'kan',
    'aceh', 'ambon', 'bali', 'bandung', 'banjarmasin', 'banten', 'blora', 'bogor',
    'cirebon', 'jambi', 'jateng', 'jember', 'kediri', 'kupang', 'lampung', 'lombok',
    'makassar', 'malang', 'manado', 'medan', 'padang', 'palembang', 'papua', 'pontianak',
    'semarang', 'solo', 'surabaya', 'tegal', 'yogyakarta', 'jakarta', 'daerah', 'kota',
    'kabupaten', 'provinsi', 'desa', 'kampung', 'kelurahan', 'kecamatan'
]

def smart_normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'([a-z])\1{2,}', r'\1', text)
    text = re.sub(
        r'\b(ya|iya|apa|mana|gimana|kenapa|kapan|siapa|berapa|tolong|dong|sih|nih|kok|udah|belum|sampe|sampai|nga|ngga|ga|gak)a+\b',
        r'\1',
        text
    )

    basic_word_map = {
        r'\byaa+\b': 'ya',
        r'\biyaa+\b': 'iya',
        r'\btu\b': 'itu',
        r'\btuh\b': 'itu',
        r'\bni\b': 'ini',
        r'\bnih\b': 'ini',
        r'\bnga\b': 'tidak',
        r'\bngga\b': 'tidak',
        r'\bnggak\b': 'tidak',
        r'\benggak\b': 'tidak',
        r'\bengga\b': 'tidak',
        r'\bgak\b': 'tidak',
        r'\bga\b': 'tidak',
        r'\bgk\b': 'tidak',
        r'\btdk\b': 'tidak',
        r'\btak\b': 'tidak',
        r'\bkagak\b': 'tidak',
        r'\bkaga\b': 'tidak',
        r'\bndak\b': 'tidak',
        r'\bnda\b': 'tidak',
        r'\bora\b': 'tidak',
        r'\bblm\b': 'belum',
        r'\bblom\b': 'belum',
        r'\bbelom\b': 'belum',
        r'\bjgn\b': 'jangan',
        r'\bsampe\b': 'sampai',
        r'\bnyampe\b': 'sampai',
        r'\bdimulai\b': 'mulai',
        r'\bdibagikan\b': 'pembagian',
        r'\bdapet\b': 'dapat',
        r'\bdpt\b': 'dapat',
        r'\bdaftar\b': 'mendaftar',
        r'\bgimana\b': 'bagaimana',
        r'\bgmna\b': 'bagaimana',
        r'\bgmn\b': 'bagaimana',
        r'\bknp\b': 'kenapa',
        r'\bkpn\b': 'kapan',
        r'\bbrp\b': 'berapa',
        r'\bdmn\b': 'di mana',
        r'\bdimana\b': 'di mana',
        r'\bkasi\b': 'kasih',
        r'\bortu\b': 'orang tua',
        r'\bortunya\b': 'orang tua',
        r'\bduit\b': 'uang',
        r'\bduitnya\b': 'uangnya',
        r'\bdananya\b': 'anggarannya',
        r'\bpekerjaan\b': 'kerja',
    }
    for pat, rep in basic_word_map.items():
        text = re.sub(pat, rep, text)

    # Pertahankan ekspresi bela MBG ("jangan nyinyir", "jangan dihentikan", "tidak setuju jika dihapuskan")
    text = re.sub(
        r'\bjangan\s+(terlalu\s+|cuma\s+|hanya\s+)?(nyinyir|suudzon|pesimis|skeptis|mencela|menghujat)\b',
        'dukung_positif_lanjutkan',
        text
    )
    text = re.sub(
        r'\b(jangan|tidak\s+setuju)\s+(sampai\s+|jika\s+|kalau\s+)?(dihentikan|dihapus|dihapuskan|dibatalkan|distop)\b',
        'dukung_positif_lanjutkan',
        text
    )

    # Mengikat Negasi + Kata Positif/Utilitas menjadi Token Kritik Eksplisit (contoh: "nga guna" -> "kritik_negasi_kontra")
    negated_pos_pattern = (
        r'\b(tidak|bukan|kurang|belum|tanpa|jangan|mana\s+ada|kata\s+siapa)\s+'
        r'(ada\s+|punya\s+|begitu\s+|terlalu\s+|pernah\s+|bisa\s+)?'
        r'(guna|gunanya|berguna|manfaat|manfaatnya|bermanfaat|bagus|baik|efektif|efisien|'
        r'tepat|jelas|penting|perlu|butuh|setuju|dukung|mendukung|layak|sehat|higienis|'
        r'solutif|masuk\s+akal|beres|becus|matang|siap|merata|transparan|membuka|buka|buat)\b'
    )
    text = re.sub(negated_pos_pattern, ' kritik_negasi_kontra ', text)

    # Deteksi Kalimat Bandingan Penolakan / Pengalihan Kebijakan ("mendingan kasih ke ortu itu lebih bermanfaat")
    has_redirection = bool(re.search(
        r'\b(mending|mendingan|daripada|ketimbang|alihkan|dialihkan|sebaiknya\s+dihentikan|'
        r'kasih\s+ke\s+orang\s+tua|kasih\s+orang\s+tua|beri\s+ke\s+orang\s+tua|diberikan\s+langsung\s+orang\s+tua|'
        r'uang\s+tunai|uangnya\s+diberikan|anggarannya\s+buat|fokus\s+pendidikan|perbaiki\s+sekolah|gaji\s+guru)\b',
        text
    ))
    if has_redirection or ('kritik_negasi_kontra' in text):
        text = re.sub(
            r'\b(jauh\s+)?(lebih\s+)?(bermanfaat|manfaat|bagus|baik|berguna|efektif|tepat|penting|solutif|berkualitas)\b',
            ' kritik_banding_kontra ',
            text
        )

    phrase_map = {
        r'\blapangan\s+kerja\b': 'membuka lapangan kerja baru',
        r'\bbuka\s+lowongan\b': 'membuka lapangan kerja baru',
        r'\bmengurangi\s+pengangguran\b': 'membuka lapangan kerja baru',
        r'\bserap\s+tenaga\s+kerja\b': 'membuka lapangan kerja baru',
        r'\buang\s+jajan\b': 'hemat uang saku membantu keluarga',
        r'\buang\s+saku\b': 'hemat uang saku membantu keluarga',
        r'\bmeringankan\s+beban\b': 'membantu keluarga kurang mampu',
        r'\bbantu\s+ekonomi\b': 'ekonomi desa bergerak positif',
        r'\bgerakkan\s+ekonomi\b': 'ekonomi desa bergerak positif',
        r'\bputar\s+ekonomi\b': 'ekonomi desa bergerak positif',
        r'\bbuang\s+buang\s+anggaran\b': 'pemborosan anggaran korupsi',
        r'\bbuang\s+anggaran\b': 'pemborosan anggaran korupsi',
        r'\bbuang\s+uang\b': 'pemborosan anggaran korupsi',
        r'\bhambur\s+anggaran\b': 'pemborosan anggaran korupsi',
    }
    for pat, rep in phrase_map.items():
        text = re.sub(pat, rep, text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("1. Membaca dan menormalisasi dataset_mbg_siap_ml.csv...")
df = pd.read_csv('dataset_mbg_siap_ml.csv')
df['clean_text'] = df['teks_model'].fillna('').apply(smart_normalize)

train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

netral_seeds = [
    "apa itu program mbg dan apa tujuannya",
    "mbg itu program apa ya sebenarnya",
    "apa itu mbg",
    "gimana cara daftar jadi mitra mbg",
    "bagaimana alur pendaftaran dapur mbg",
    "syarat mendaftar jadi penyedia makanan mbg apa saja",
    "kapan program mbg mulai berjalan",
    "kapan mbg sampai ke daerah kami",
    "apakah anak sma dan smk juga dapat jatah makan bergizi gratis",
    "berapa anggaran mbg per porsi untuk anak sekolah",
    "jadwal pembagian mbg jam berapa biasanya",
    "menu mbg hari ini apa saja ya untuk anak sekolah",
    "di mana lokasi kantor bgn atau sppg terdekat",
    "tolong info jadwal dan alur sosialisasi mbg",
    "apakah ada kontak admin atau call center resmi mbg",
    "bagaimana mekanisme pengawasan kebersihan makanan mbg",
    "siapa saja yang berhak menerima bantuan mbg ini",
    "apakah siswa madrasah dan pesantren juga dapat",
    "bagaimana cara cek status sekolah penerima mbg",
    "hari ini ada sosialisasi mbg di kantor",
    "berapa porsi jatah makanan per siswa",
    "kapan pembagian mbg mulai berjalan"
]

pro_seeds = [
    "mbg bisa membuka lapangan kerja baru bagi masyarakat",
    "program ini bagus karena membuka lapangan pekerjaan dan menggerakkan ekonomi",
    "keren banget inovasi program makan bergizi gratis ini",
    "mantap sekali program ini sangat membantu orang tua murid",
    "luar biasa inisiatif mbg gizi anak-anak jadi terjamin",
    "sangat setuju dan mendukung penuh kelanjutan program mbg",
    "alhamdulillah program mbg sangat bermanfaat bagi masyarakat",
    "terima kasih pemerintah anak kami jadi sehat dan semangat sekolah",
    "kebijakan yang sangat solutif dan tepat sasaran",
    "top banget programnya semoga terus berlanjut ke depannya",
    "kami sangat terbantu dengan adanya makan siang gratis ini",
    "semoga sukses selalu program mbg demi masa depan anak bangsa",
    "hemat uang saku membantu keluarga kurang mampu",
    "petani lokal dan umkm ikut terbantu karena dapur mbg"
]

kontra_seeds = [
    "program tidak berguna mendingan kasih ke orang tua siswa lebih bermanfaat",
    "mending uangnya buat gaji guru dan perbaiki sekolah rusak daripada mbg",
    "program tidak jelas dan tidak ada gunanya buang anggaran negara",
    "alihkan saja dana mbg untuk beasiswa dan pendidikan gratis lebih tepat",
    "percuma makan gratis kalau kualitasnya buruk dan bikin anak keracunan"
]

aug_rows = []
for s in netral_seeds * 18:
    aug_rows.append({'clean_text': smart_normalize(s), 'label_id': 1})
for s in pro_seeds * 10:
    aug_rows.append({'clean_text': smart_normalize(s), 'label_id': 2})
for s in kontra_seeds * 12:
    aug_rows.append({'clean_text': smart_normalize(s), 'label_id': 0})

aug_df = pd.DataFrame(aug_rows)
train_balanced = pd.concat([train_df[['clean_text', 'label_id']], aug_df], ignore_index=True)

print("2. Mengekstraksi fitur TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    stop_words=FALSE_BIAS_STOPWORDS
)

X_train = vectorizer.fit_transform(train_balanced['clean_text'])
y_train = train_balanced['label_id']

X_test = vectorizer.transform(test_df['clean_text'])
y_test = test_df['label_id']

print("3. Melatih Multi-Class Ensemble (Support Vector Machine + Naive Bayes)...")
mnb = MultinomialNB(alpha=0.5, fit_prior=False)
cnb = ComplementNB(alpha=0.5, fit_prior=False)
svc = CalibratedClassifierCV(
    LinearSVC(class_weight='balanced', dual=False, C=0.8, random_state=42),
    cv=3
)

model = VotingClassifier(
    estimators=[
        ('multinomial_nb', mnb),
        ('complement_nb', cnb),
        ('linear_svc', svc)
    ],
    voting='soft',
    weights=[1, 1, 2]
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
target_names = ['kontra', 'netral', 'pro']

acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro')
rec_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

print("=" * 60)
print(f"Akurasi Uji Multi-Class Ensemble: {acc * 100:.2f}% | Macro F1: {f1_macro * 100:.2f}%")
print("=" * 60)

metrics_data = {
    'accuracy': acc,
    'precision': prec_macro,
    'recall': rec_macro,
    'f1_score': f1_macro,
    'confusion_matrix': cm.tolist(),
    'classification_report': clf_report,
    'labels': target_names,
    'total_train': len(train_df),
    'total_test': len(test_df),
    'class_distribution': df['label'].value_counts().to_dict()
}

joblib.dump(model, 'model_sentimen_mbg.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(metrics_data, 'metrics_evaluasi.pkl')
print("Model baru berhasil disimpan!")