# Information-Retrieval-WebMining
Proyek web mining yang memanfaatkan TF IDF untuk melakukan information retrieval dengan memanfaatkan matching score atau cosine similarity untuk mendapatkan best match dari dataset yang diberikan

Aplikasi ini didasarkan pada git https://github.com/williamscott701/Information-Retrieval/tree/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score

Input yang diberikan ketika menjalankan aplikasi ini adalah:
- Query yang diinginkan
- Jumlah dokumen yang akan dihasilkan

Output yang dihasilkan setelah memproses:
- File HTML sesuai jumlah dokumen yang diminta
- Judul dokumen yang menjadi best matches

Dataset yang digunakan adalah dataset dalam bentuk JSON yaitu:
- raw.json
  - url
  - html
- processed.json
  - title
  - content
  - published_at
  - fetched_at
  - url
  - topic

Prasyarat:
- Terdapat folder "HTML" untuk menampung file HTML yang dibuat ("folder git/HTML")
- Dataset diletakkan pada ("folder git/folder dataset individu pertama/raw.json" & "folder git/folder dataset individu pertama/processed.json")
- File JSON harus dapat dibuka menggunakan library json (import json) dengan json.load("namafile.json"), atau akan keluar error

Catatan:
- Dataset yang digunakan untuk membuat aplikasi ini dapat didownload pada https://drive.google.com/file/d/135E57IGhepBVbDEi2pv9h65JCQEbYHdu/view?usp=sharing 
- Aplikasi ini dibuat untuk memenuhi proyek Web Mining Semester Genap 2019/2020 di iSTTS