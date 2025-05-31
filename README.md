# Tugas Besar Mata Kuliah Digital Processing Signal (IF3024)

## Dosen Pengampu: **Martin Clinton Tosima Manullang, S.T., M.T..**

# **RPPG & Resp Signal**

## **Anggota Kelompok**

---

| **Nama**                     | **NIM**   | **ID GITHUB**                                                       |
| ---------------------------- | --------- | ------------------------------------------------------------------- |
| Winnerson Laia               | 121140121 | <a href="https://github.com/Winnerson-121140121">github winner </a> |
| Muhammad Qaessar Kartadilaga | 121140119 | <a href="https://github.com/121140119Qaessar">github qaessaar</a>   |

---

## **Deskripsi Proyek**

Proyek ini merupakan tugas besar dari mata kuliah Pengolahan Sinyal Digital IF(3024) yang bertujuan untuk mengekstraksi sinyal respirasi dan sinyal remote-photoplethysmography (rPPG) dari video input.
Untuk mendapatkan sinyal respirasi, program memanfaatkan pose-landmarker dari MediaPipe guna mendeteksi gerakan bahu yang terjadi selama proses pernapasan.
Sementara itu, sinyal rPPG diperoleh menggunakan face-detector dari MediaPipe dan algoritma Plane Orthogonal-to-Skin (POS), yang menganalisis perubahan warna pada wajah pengguna untuk menghitung denyut jantung secara non-kontak.

---

## **Teknologi yang Digunakan**

Berikut adalah teknologi dan alat yang digunakan dalam proyek ini:

| Logo                                                                                                                           | Nama Teknologi | Fungsi                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------- | -------------------------------------------------------------------------------- |
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="60">            | Python         | Bahasa pemrograman utama untuk pengembangan filter.                              |
| <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" alt="VS Code Logo" width="60"> | VS Code        | Editor teks untuk mengedit skrip secara efisien dengan dukungan ekstensi Python. |

---

## **Library yang Digunakan**

Berikut adalah daftar library Python yang digunakan dalam proyek ini, beserta fungsinya:

| **Library**                    | **Fungsi**                                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| `cv2`                          | Digunakan untuk menangkap gambar dari kamera dan memproses gambar secara langsung.                 |
| `mediapipe`                    | Digunakan untuk mendeteksi landmark wajah, seperti posisi hidung, untuk mendeteksi gerakan kepala. |
| `scipy`, `numpy`, `matplotlib` | Digunakan untuk bahan oprasi pembuatan program                                                     |

---

## **Fitur**

### **1. Resp Signal (Remote Photoplethysmography)**

- Bagian ini menggunakan metode Pose Detection dan Optical Tracking dengan Tracking Failure untuk mendeteksi gerakan bahu dan dada untuk mensimmulasikan gerakan pernafasan.

### **2. RPPG (Remote Photoplethysmography)**

- Bagian ini rPPG menggunakan teknologi visi komputer untuk mengekstrak informasi tentang perubahan penyerapan cahaya pada kulit wajah.
  Sinyal yang didapat akan di proses menjadi sinyal detak jantung.

---

## Logbook

### Tanggal 27 Mei 2025

- Inisialisasi github repo dan github project management untuk manajemen tugas besar "Digital Signal Processing"
- Menambahkan folder fungsi, models, dan sampel.
- Memulai pembuatan laporan.

### Tanggal 30 Mei 2025

- Menambahkan file main.py
- Menambahkan modul pemrosesan_respirasi
- Menambahkan modul pemrosesan_rppg
- Menambahkan file environtment.yml dan requirement.txt

### Tanggal 31 Mei 2025

- Menperbaharui readme dan menambahkan file report

## Cara Menjalankan Program

Dengan asumsi bahwa Anda sudah mempunyai environment manager seperti conda. maka buat environment baru seperti ini. Clone / fork lalu jalankan perintah ini.

### 1. Dengan environment.yml

```yaml
conda env create -f environment.yml
```

Lalu bukan environment yang sudah dibuat sebelumnya dengan.

```yaml
conda activate TUBES_BESAR_DSP
```

Lalu jalankan perintah ini untuk menjalankan program.

```yaml
python main.py
```

### 2. Dengan requirements.txt

Atau jika Anda mengalami masalah dalam menggunakan environment.yml, anda bisa menggunakan requirements.txt. Jalankan perintah ini.

```yaml
conda create -n TUBES_BESAR_DSP python=3.10.16
```

Lalu buka environment yang sudah dibuat sebelumnya dengan

```yaml
conda activate TUBES_BESAR_DSP
```

Jalankan perintah ini untuk menginstall library yang dibutuhkan.

```yaml
pip install -r requirements.txt
```

Lalu jalankan perintah ini untuk menjalankan program.

```yaml
python main.py
```
