# Information-Retrieval-WebMining
Proyek web mining yang memanfaatkan TF IDF untuk melakukan information retrieval dengan memanfaatkan matching score atau cosine similarity untuk mendapatkan best match dari dataset yang diberikan

Aplikasi ini didasarkan pada git https://github.com/williamscott701/Information-Retrieval/tree/master/2.%20TF-IDF%20Ranking%20-%20Cosine%20Similarity%2C%20Matching%20Score\

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
