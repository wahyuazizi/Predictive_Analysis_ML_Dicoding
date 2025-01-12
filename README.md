# Laporan Proyek Machine Learning - Wahyu Azizi
___

# Domain Proyek
___

Peningkatan emisi karbon dioksida (CO2) dari kendaraan bermotor menjadi salah satu penyumbang utama perubahan iklim global. Hal ini disebabkan oleh pertumbuhan jumlah kendaraan yang pesat, terutama di negara-negara berkembang seperti Indonesia. Kendaraan bermotor, terutama yang menggunakan bahan bakar fosil, menghasilkan emisi CO2 melalui proses pembakaran. Penelitian menunjukkan bahwa sektor transportasi berkontribusi signifikan terhadap total emisi gas rumah kaca. Di Indonesia, emisi dari transportasi diperkirakan mencapai 85% dari total pencemaran udara, dengan kendaraan pribadi dan sepeda motor sebagai penyumbang terbesar [[1](https://www.semanticscholar.org/paper/Kajian-Emisi-CO2-Dari-Kendaraan-Bermotor-Di-Kampus-Anggraeni/214a377555002c31829c59dbf1dea7cfb396ef8d)].  
Sebuah studi di Jalan Raya Kemantren, Sidoarjo, menunjukkan bahwa potensi emisi CO2 dari sepeda motor mencapai 67.568,26 g dalam 30 menit per kilometer, sedangkan mobil menghasilkan sekitar 63.335,30 g pada periode yang sama[[2](https://www.semanticscholar.org/paper/Analisis-Potensi-Emisi-CO2-Oleh-Berbagai-Jenis-di-Sudarti-Yushardi/81d79cda27154bdf4c54548423a878659b872df5)]. Selain itu, kawasan industri seperti SIER di Surabaya juga mengalami peningkatan emisi CO2 akibat tingginya kepadatan kendaraan dan kemacetan[[3](https://www.semanticscholar.org/paper/Analisis-Besaran-Emisi-Gas-CO2-Kendaraan-Bermotor-Kusumawardani-Navastara/54310e88f91c19e3519403d20b70adf299d55a3b)][[4](https://www.semanticscholar.org/paper/Arahan-Penyediaan-Ruang-Terbuka-Hijau-Dalam-Emisi-Kusumawardani/52438bef01b4397f8f43cc90d4a3a8dafc5edea6)]. Peningkatan emisi CO2 berkontribusi pada efek rumah kaca, yang berujung pada pemanasan global. Kenaikan suhu akibat peningkatan konsentrasi gas rumah kaca dapat menyebabkan perubahan iklim yang ekstrem, termasuk cuaca yang tidak menentu dan bencana alam[[5](https://www.semanticscholar.org/paper/ANALISIS-SHIFTING-PENGGUNAAN-MODA-KENDARAAN-KE-API-Amelia-Samadikun/9b4974abd17e7a013948ebb8f2ca4cf11b924e60)]. Penelitian juga menunjukkan bahwa daerah dengan kepadatan transportasi tinggi mengalami penurunan kualitas udara dan peningkatan suhu[[3](https://www.semanticscholar.org/paper/Analisis-Besaran-Emisi-Gas-CO2-Kendaraan-Bermotor-Kusumawardani-Navastara/54310e88f91c19e3519403d20b70adf299d55a3b)].  
Kualitas udara yang buruk diakibatkan oleh peningkatan jumlah kendaraan bermotor yang menghasilkan emisi gas berbahaya seperti karbon dioksida (CO2), nitrogen oksida (NOx), dan partikel halus (PM). Di DKI Jakarta, misalnya, kebijakan uji emisi diharapkan dapat mengurangi polusi, namun tantangan dalam pengawasan dan pertumbuhan jumlah kendaraan tetap menjadi masalah besar[[6](https://www.semanticscholar.org/paper/Analisis-Kebijakan-Pengendalian-Polusi-melalui-Uji-Paradizsa/72478900647357dab65f36a62ac558041e1dff34)]. Penelitian menunjukkan bahwa emisi dari kendaraan pribadi berkontribusi signifikan terhadap pencemaran udara di daerah dengan kepadatan lalu lintas tinggi, seperti Kota Semarang[[7](https://www.semanticscholar.org/paper/ESTIMASI-EMISI-PENCEMAR-UDARA-KONVENSIONAL-(SOx%2C-DI-Buanawati-Huboyo/1aa8e1dc1be0e8e43c33b4a19a806625c8f01d15)].  
Data science memainkan peran penting dalam memahami dan menganalisis pola emisi dari kendaraan bermotor. Dengan memanfaatkan teknik analisis data, machine learning, dan pemodelan statistik, para peneliti dan pembuat kebijakan dapat memperoleh wawasan yang lebih baik mengenai sumber emisi, tren, dan dampaknya terhadap lingkungan.

# Business Understanding
___

## Problem Statement

1. Bagaimana cara memprediksi emisi CO2 kendaraan berdasarkan karakteristik seperti ukuran mesin, jenis bahan bakar, dan konsumsi bahan bakar?
2. Bagaimana hubungan variabel seperti tipe atau ukuran mesin, jenis bahan bakar, dan efisiensi bahan bakar terhadap tingkat emisi CO2 kendaraan?

## Goals

1. Membangun model prediktif untuk memprediksi emisi CO2 kendaraan berdasarkan dataset yang tersedia.
2. Mengidentifikasi hubungan antara variabel seperti tipe atau ukuran mesin, jenis bahan bakar, dan konsumsi bahan bakar dan variabel lainnya terhadap emisi CO2.

## Solutions

1. Pemanfaatan Dataset
Dataset mencakup berbagai variabel kendaraan seperti:
- Spesifikasi kendaraan: merek, model, ukuran mesin, jumlah silinder, jenis bahan bakar, dan jenis transmisi.
- Efisiensi bahan bakar: konsumsi bahan bakar di area perkotaan, jalan raya, dan kombinasi dalam liter per 100 km atau mil per galon (mpg).
- Target prediksi: emisi CO2 dalam gram per kilometer.
2. Penggunaan Algoritma Machine Learning
Untuk membangun model prediksi emisi CO2, digunakan tiga algoritma:

- ```K-Nearest Neighbors (KNN)```:
  Memanfaatkan kemiripan antara kendaraan berdasarkan spesifikasi dan konsumsi bahan bakar. Cocok untuk data dengan pola non-linear, meskipun memerlukan tuning parameter seperti jumlah tetangga (k) agar optimal.
- ```Random Forest```:  
    Membantu memahami faktor mana yang paling memengaruhi emisi, seperti kapasitas mesin atau jenis bahan bakar. Memberikan hasil prediksi yang stabil dengan menangani dataset yang kompleks dan tidak seimbang.
- ```Boosting```:  
    Digunakan untuk meningkatkan akurasi dengan mengatasi kesalahan prediksi pada model sebelumnya. Mampu menangani interaksi kompleks antara fitur, cocok untuk dataset ini yang melibatkan banyak variabel interdependen.


# Data Understanding
___

Dataset ini berisi informasi tentang spesifikasi kendaraan, konsumsi bahan bakar, dan emisi CO2 yang dikumpulkan untuk menganalisis dampak lingkungan dari kendaraan serta memprediksi emisi CO2 menggunakan model regresi. Dataset ini dirancang untuk mendukung pendekatan **Simple Linear Regression (SLR)** dan **Multiple Linear Regression (MLR)** dalam proyek machine learning.

Tautannya dapat diakses di sini:  
[Vehicle CO2 Emissions Dataset](https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset)

## Data Loading

Sebelum melakukan eksplorasi data, terlebih dahulu kita load data, seperti berikut:






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Vehicle Class</th>
      <th>Engine Size(L)</th>
      <th>Cylinders</th>
      <th>Transmission</th>
      <th>Fuel Type</th>
      <th>Fuel Consumption City (L/100 km)</th>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <th>Fuel Consumption Comb (L/100 km)</th>
      <th>Fuel Consumption Comb (mpg)</th>
      <th>CO2 Emissions(g/km)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACURA</td>
      <td>ILX</td>
      <td>COMPACT</td>
      <td>2.0</td>
      <td>4</td>
      <td>AS5</td>
      <td>Z</td>
      <td>9.9</td>
      <td>6.7</td>
      <td>8.5</td>
      <td>33</td>
      <td>196</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACURA</td>
      <td>ILX</td>
      <td>COMPACT</td>
      <td>2.4</td>
      <td>4</td>
      <td>M6</td>
      <td>Z</td>
      <td>11.2</td>
      <td>7.7</td>
      <td>9.6</td>
      <td>29</td>
      <td>221</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACURA</td>
      <td>ILX HYBRID</td>
      <td>COMPACT</td>
      <td>1.5</td>
      <td>4</td>
      <td>AV7</td>
      <td>Z</td>
      <td>6.0</td>
      <td>5.8</td>
      <td>5.9</td>
      <td>48</td>
      <td>136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACURA</td>
      <td>MDX 4WD</td>
      <td>SUV - SMALL</td>
      <td>3.5</td>
      <td>6</td>
      <td>AS6</td>
      <td>Z</td>
      <td>12.7</td>
      <td>9.1</td>
      <td>11.1</td>
      <td>25</td>
      <td>255</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACURA</td>
      <td>RDX AWD</td>
      <td>SUV - SMALL</td>
      <td>3.5</td>
      <td>6</td>
      <td>AS6</td>
      <td>Z</td>
      <td>12.1</td>
      <td>8.7</td>
      <td>10.6</td>
      <td>27</td>
      <td>244</td>
    </tr>
  </tbody>
</table>
</div>



Dari hasil diatas diperoleh:
* total record sebanyak ```7385```
* terdapat ```12 variabel```

## Exploratory Data Analysis

### EDA - Deskripsi Variabel

Berikut detail dari tiap-tiap variabel yang ada untuk memahami data yang kita proses

| Nama Kolom                          | Deskripsi                                                                 |
|-------------------------------------|---------------------------------------------------------------------------|
| Make                                | Merek atau produsen kendaraan (contoh: Toyota, Ford, BMW).               |
| Model                               | Model spesifik kendaraan.                                                |
| Vehicle Class                       | Klasifikasi kendaraan berdasarkan ukuran dan penggunaan (contoh: SUV, Sedan). |
| Engine Size(L)                      | Volume kapasitas mesin dalam liter.                                      |
| Cylinders                           | Jumlah silinder pada mesin kendaraan.                                    |
| Transmission                        | Jenis transmisi kendaraan (contoh: Otomatis, Manual).                    |
| Fuel Type                           | Jenis bahan bakar yang digunakan oleh kendaraan.                         |
| Fuel Consumption City (L/100 km)    | Konsumsi bahan bakar di area perkotaan dalam liter per 100 kilometer.    |
| Fuel Consumption Hwy (L/100 km)     | Konsumsi bahan bakar di jalan raya dalam liter per 100 kilometer.        |
| Fuel Consumption Comb (L/100 km)    | Konsumsi bahan bakar gabungan (rata-rata kota dan jalan raya) dalam liter per 100 kilometer. |
| Fuel Consumption Comb (mpg)         | Konsumsi bahan bakar gabungan dalam mil per galon (mpg).                 |
| CO2 Emissions(g/km)                 | Emisi karbon dioksida per kilometer (target prediksi).                   |

### Penjelasan Fuel Type:
- **X**: Bensin biasa (regular gasoline)  
- **Z**: Bensin premium (premium gasoline)  
- **D**: Diesel  
- **E**: E85 (campuran bensin dan etanol)  
- **N**: Gas alam (natural gas)  


**Tipe Variabel**



    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7385 entries, 0 to 7384
    Data columns (total 12 columns):
     #   Column                            Non-Null Count  Dtype  
    ---  ------                            --------------  -----  
     0   Make                              7385 non-null   object 
     1   Model                             7385 non-null   object 
     2   Vehicle Class                     7385 non-null   object 
     3   Engine Size(L)                    7385 non-null   float64
     4   Cylinders                         7385 non-null   int64  
     5   Transmission                      7385 non-null   object 
     6   Fuel Type                         7385 non-null   object 
     7   Fuel Consumption City (L/100 km)  7385 non-null   float64
     8   Fuel Consumption Hwy (L/100 km)   7385 non-null   float64
     9   Fuel Consumption Comb (L/100 km)  7385 non-null   float64
     10  Fuel Consumption Comb (mpg)       7385 non-null   int64  
     11  CO2 Emissions(g/km)               7385 non-null   int64  
    dtypes: float64(4), int64(3), object(5)
    memory usage: 692.5+ KB


dari deskripsi variabel dan informasi di atas, dapat diperoleh bahwa tidak ada missing value dan tidak terdapat masalah pada tipe data dalam dataset. Selanjutnya untuk lebih detailnya periksa total missing value dan jika terdapat duplikasi data.

### EDA - Cleaning Data

#### Identifikasi *Missing Value* & Duplikasi Data





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daftar Missing Value:</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Make</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Model</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Vehicle Class</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Engine Size(L)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Cylinders</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Transmission</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Fuel Type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Fuel Consumption City (L/100 km)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Fuel Consumption Comb (L/100 km)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Fuel Consumption Comb (mpg)</th>
      <td>0</td>
    </tr>
    <tr>
      <th>CO2 Emissions(g/km)</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






    Jumlah duplikat: 1103


Berdasarkan informasi diatas, dataset tersebut tidak memiliki missing value sama sekali, akan tetapi memiliki duplicate value yang lumayan banyak yaitu sebanyak ```1103``` data yang terduplikasi, sehingga harus dihilangkan.

#### Identifikasi Outliers




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Engine Size(L)</th>
      <th>Cylinders</th>
      <th>Fuel Consumption City (L/100 km)</th>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <th>Fuel Consumption Comb (L/100 km)</th>
      <th>Fuel Consumption Comb (mpg)</th>
      <th>CO2 Emissions(g/km)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7385.000000</td>
      <td>7385.000000</td>
      <td>7385.000000</td>
      <td>7385.000000</td>
      <td>7385.000000</td>
      <td>7385.000000</td>
      <td>7385.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.160068</td>
      <td>5.615030</td>
      <td>12.556534</td>
      <td>9.041706</td>
      <td>10.975071</td>
      <td>27.481652</td>
      <td>250.584699</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.354170</td>
      <td>1.828307</td>
      <td>3.500274</td>
      <td>2.224456</td>
      <td>2.892506</td>
      <td>7.231879</td>
      <td>58.512679</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.900000</td>
      <td>3.000000</td>
      <td>4.200000</td>
      <td>4.000000</td>
      <td>4.100000</td>
      <td>11.000000</td>
      <td>96.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>10.100000</td>
      <td>7.500000</td>
      <td>8.900000</td>
      <td>22.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>12.100000</td>
      <td>8.700000</td>
      <td>10.600000</td>
      <td>27.000000</td>
      <td>246.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.700000</td>
      <td>6.000000</td>
      <td>14.600000</td>
      <td>10.200000</td>
      <td>12.600000</td>
      <td>32.000000</td>
      <td>288.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.400000</td>
      <td>16.000000</td>
      <td>30.600000</td>
      <td>20.600000</td>
      <td>26.100000</td>
      <td>69.000000</td>
      <td>522.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### **Penjelasan Statistik Deskriptif**

**1. Engine Size (L)**:
- **Rata-rata** ukuran mesin adalah **3.16 L**, dengan **median** (nilai tengah) sebesar **3.00 L**.
- Ukuran mesin terkecil adalah **0.90 L**, sedangkan yang terbesar adalah **8.40 L**.
- **Standar deviasi** sebesar **1.37 L** menunjukkan bahwa ukuran mesin memiliki variasi yang cukup besar.
- **Kuartil pertama (Q1)** sebesar **2.00 L** dan **kuartil ketiga (Q3)** sebesar **3.70 L** menunjukkan bahwa 50% kendaraan memiliki ukuran mesin antara **2.00 L** dan **3.70 L**.

**2. Cylinders (Jumlah Silinder)**:
- Kendaraan memiliki rata-rata **5.62 silinder**, dengan **median** sebesar **6 silinder**.
- Kendaraan dengan jumlah silinder terkecil memiliki **3 silinder**, sementara yang terbesar memiliki **16 silinder**.
- Sebagian besar kendaraan (50%) memiliki jumlah silinder antara **4** dan **6**.

**3. Fuel Consumption City (L/100 km)**:
- **Rata-rata konsumsi bahan bakar di kota** adalah **12.61 L/100 km**, dengan nilai **median** sebesar **12.10 L/100 km**.
- Konsumsi terkecil adalah **4.20 L/100 km**, sementara yang terbesar mencapai **30.60 L/100 km**.
- Nilai **kuartil pertama (Q1)** sebesar **10.10 L/100 km** dan **kuartil ketiga (Q3)** sebesar **14.70 L/100 km** menunjukkan bahwa sebagian besar kendaraan berada di antara rentang ini.

**4. Fuel Consumption Hwy (L/100 km)**:
- **Rata-rata konsumsi bahan bakar di jalan raya** adalah **9.07 L/100 km**, dengan median sebesar **8.70 L/100 km**.
- Konsumsi bahan bakar minimum adalah **4.00 L/100 km**, sementara maksimum adalah **20.60 L/100 km**.
- Sebagian besar kendaraan memiliki konsumsi bahan bakar jalan raya antara **7.50 L/100 km** (Q1) dan **10.30 L/100 km** (Q3).

**5. Fuel Consumption Comb (L/100 km)**:
- **Rata-rata konsumsi bahan bakar kombinasi** adalah **11.02 L/100 km**, dengan nilai median sebesar **10.60 L/100 km**.
- Rentang konsumsi adalah dari **4.10 L/100 km** hingga **26.10 L/100 km**.
- Sebagian besar kendaraan memiliki konsumsi bahan bakar kombinasi antara **8.90 L/100 km** (Q1) dan **12.70 L/100 km** (Q3).

**6. Fuel Consumption Comb (mpg)**:
- **Rata-rata efisiensi bahan bakar dalam miles per gallon (mpg)** adalah **27.41 mpg**, dengan median sebesar **27 mpg**.
- Kendaraan paling tidak efisien memiliki efisiensi **11 mpg**, sedangkan kendaraan paling efisien mencapai **69 mpg**.
- Sebagian besar kendaraan memiliki efisiensi antara **22 mpg** (Q1) dan **32 mpg** (Q3).

**7. CO2 Emissions (g/km)**:
- **Rata-rata emisi CO2** adalah **251.16 g/km**, dengan nilai median sebesar **246 g/km**.
- Kendaraan dengan emisi CO2 terendah menghasilkan **96 g/km**, sedangkan kendaraan dengan emisi tertinggi menghasilkan **522 g/km**.
- Sebagian besar kendaraan memiliki emisi antara **208 g/km** (Q1) dan **289 g/km** (Q3).





    
![png](gambar_files/gambar_36_0.png)
    
Nilai yang jauh di luar rentang normal, seperti ```Engine Size```, ```Cylinder```, konsumsi bahan bakar kombinasi yang mencapai **26.10 L/100 km** atau emisi CO2 sebesar **522 g/km**, dan variabel mungkin memerlukan perhatian lebih karena berpotensi menjadi *outlier*.


### EDA - Univariate Analysis

#### Word Cloud




    
![png](gambar_files/gambar_39_0.png)
    


Variabel Model memiliki value sangat beragam sehingga visualisasi word cloud adalah yang cocok. Dimana Model sepet ```AWD```, ```COOPER```, ```TYPE```, ```FFV```, ```ROVER``` lebih banyak muncul dalam data

#### Categorical Features




    
![png](gambar_files/gambar_42_0.png)
    


Berikutnya, distribusi tiap variabel kategori.  
- Distribusi ```Make```  ada vehicle dari ```FORD, BMW, CHEVROLET``` merupakan tiga teratas yang paling banyak digunakan dan ```LAMBORGHINI``` yang paling rendah.
- Pada variabel ```Vehicle Class``` ada class```SUV-SMALL, MID-SIZE, COMPACT``` yang merupakan top three yang paling banyak digunakan dan class ```VAN-CARGO``` adalah yang paling sedikit.
- Untuk variabel  ```Transmission``` terdapat ```AS6, AS8, M6, A6``` yang merupakan empat teratas untuk transmisi yang paling banyak digunakan dan transmisi ```AV10, AM5, AS4, AM9``` merupakan yang paling rendah.
- Dan pada variabel ```Fuel Type```, Jenis ```X (Regular) dan Z (Premium)``` adalah tipe bahan bakar yang paling umum digunakan, type ```E (E85 Campuran Bensin dan Etanol) dan D (Diesel)``` dengan rentang 147-223, sedangkan ```N (Natural Gas)``` hanya satu dalam data.

#### Numerical Features



    
![png](gambar_files/gambar_45_0.png)
    


**Distribusi fitur numerik**  
- ```Engine Size(L)``` dengan rentang 2-4 merupakan size yang paling umum digunakan
- ```Cylinders``` dengan value ```4,6,8``` adalah cylinder yang paling banyak digunakan pada kendaraan.
- Pada ```Fuel Consumption City L/100Km``` paling banyak dihabiskan diantara rentang 10-13L per 100 Km
- Untuk ```Fuel Consumption Hwy L/100Km``` paling banyak dihabiskan dengan rentang 7-9L per 100 Km
- Fitur ```Fuel Consumption Comb L/100Km``` dengan rentang 8-12L paling banyak dihabiskan per 100 Km
- Pada ```Fuel Consumption Comb mpg``` paling banyak menjangkau diantara rentang 20-35 miles per galon
- Dan pada fitur ```CO2 Emissions g/km``` sumbangan emisi diantara 200-250 paling banyak

### EDA - Multivariate Analysis

#### Categorical Features



    
![png](gambar_files/gambar_50_0.png)
    



    
![png](gambar_files/gambar_50_1.png)
    



    
![png](gambar_files/gambar_50_2.png)
    



    
![png](gambar_files/gambar_50_3.png)
    


**Distribusi fitur kategori terhadap CO2 Emissions**  

- Fitur ```Make``` dengan penyumbang emisi terbanyak adalah ```BUGATTI, ROLLS-ROYCE, SRT, LAMBORGHINI, ASTON MARTIN, BENTLEY, dan MASERATI``` akan tetapi penggunaannya sangat-sangat rendah, bisa dilihat pada barplot horizontal di fase *Univariate Analysis*, dan rata-rata menyumbang emisi rentang 200-250 g/km
- Penyumbang emisi terbanyak pada fitur ```Vehicle Class``` adalah ```VAN-CARGO, dan VAN-PASSENGER``` tapi merupakan class yang paling sedikit digunakan. sedangkan untuk class yang paling banyak digunakan yakni ```SUV-SMALL, MID-SIZE, dan COMPACT``` menyumbang emisi diantara rentang 200-250 g/km
- Selanjutnya jenis ```Transmission``` yang paling banyak menyumbang emisi CO2 adalah jenis ```A10, A7, dan A5``` tapi merupakan jenis tranmisi yang tergolong rendah untuk penggunaannya. Untuk Transmisi yang paling sering digunakan yakni ```AS6, AS8, dan M6``` menyumbang emisi dengan rentang 200-250 g/km
- Rata-rata semua ```Fuel Type``` menyumbang emisi yang tinggi, akan tetapi type ```N``` belum bisa disimpulkan karna data yang ada hanya satu dalam dataset.

#### Numerical Features




    
![png](gambar_files/gambar_53_0.png)
    


Relasi antara fitur numerik dengan fitur target yakni ```CO2 Emissions``` positif kecuali fitur ```Fuel Consumption(mpg)```, ini karna metode perhitungan miles per galon menghitung total jarak yang ditempuh per galonnya, semakin tinggi nilai tempuhnya maka semakin sedikit emisi yang dikeluarkan. Dari fitur ```Engine Size (L)``` semakin besar kapasitas mesin maka emisi yang dihasilkan semakin besar. Pada fitur ```Cylinders``` juga menunjukkan relasi yang positif dengan cylinder tertinggi menghasilkan emisi yang tinggi juga. untuk fitur ```Consumption``` yang menggunakan ukuran L/km menunjunkan relasi yang positif bahwa meningkatnya penggunaan tentu menghasilkan emisi yang berlebihan.
Nilai relasi rentang -1 hingga +1 bisa dilihat dalam hetmap berikut.





    
![png](gambar_files/gambar_55_0.png)
    


# Data Preparation
___

## Menangani Duplikasi Data dan Outliers

**Membersihkan Duplikasi Data**


    Jumlah duplikat setelah pembersihan: 0


##### **Menangani Outlier Menggunakan Metode IQR**

Outlier adalah nilai yang berada jauh di luar rentang normal data. Salah satu cara yang efektif untuk mengidentifikasi dan menangani outlier adalah menggunakan **Metode IQR (Interquartile Range)**. Berikut adalah langkah-langkah dan cara melakukannya:

1. **Definisi IQR**
Interquartile Range (IQR) adalah rentang antara kuartil ketiga (Q3) dan kuartil pertama (Q1), yang merepresentasikan 50% data tengah.

2. **Identifikasi Outlier**
Outlier didefinisikan sebagai nilai yang berada di bawah atau di atas batas berikut:
- **Batas Bawah (Lower Bound)**
- **Batas Atas (Upper Bound)**

Nilai yang:
- Kurang dari **Lower Bound** dianggap sebagai *outlier bawah*.
- Lebih dari **Upper Bound** dianggap sebagai *outlier atas*.

3. **Langkah-langkah Penanganan**  
$$
\text{IQR} = Q3 - Q1
$$

Batas bawah (Lower Bound):
$$
\text{Lower Bound} = Q1 - 1.5 \times \text{IQR}
$$

Batas atas (Upper Bound):
$$
\text{Upper Bound} = Q3 + 1.5 \times \text{IQR}
$$








    (5816, 12)



Setelah pembersihan, data saat ini memiliki shape **```(5816, 12)```**






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Engine Size(L)</th>
      <th>Cylinders</th>
      <th>Fuel Consumption City (L/100 km)</th>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <th>Fuel Consumption Comb (L/100 km)</th>
      <th>Fuel Consumption Comb (mpg)</th>
      <th>CO2 Emissions(g/km)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5816.000000</td>
      <td>5816.000000</td>
      <td>5816.000000</td>
      <td>5816.000000</td>
      <td>5816.000000</td>
      <td>5816.000000</td>
      <td>5816.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.014254</td>
      <td>5.391678</td>
      <td>12.233253</td>
      <td>8.831327</td>
      <td>10.702975</td>
      <td>27.664718</td>
      <td>246.004814</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.215559</td>
      <td>1.517327</td>
      <td>2.858852</td>
      <td>1.864839</td>
      <td>2.379066</td>
      <td>6.043525</td>
      <td>50.208692</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.900000</td>
      <td>3.000000</td>
      <td>5.600000</td>
      <td>4.500000</td>
      <td>6.000000</td>
      <td>16.000000</td>
      <td>128.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>7.400000</td>
      <td>8.900000</td>
      <td>23.000000</td>
      <td>207.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>11.900000</td>
      <td>8.600000</td>
      <td>10.400000</td>
      <td>27.000000</td>
      <td>242.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.600000</td>
      <td>6.000000</td>
      <td>14.100000</td>
      <td>9.900000</td>
      <td>12.300000</td>
      <td>32.000000</td>
      <td>281.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.200000</td>
      <td>8.000000</td>
      <td>21.300000</td>
      <td>14.500000</td>
      <td>18.100000</td>
      <td>47.000000</td>
      <td>407.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Encoding Fitur Kategori

Dikarenakan variabel ```Model``` memiliki value yang sangat beragam, maka kita menggunakan teknik Frequency Encoding. Frequency Encoding adalah salah satu metode encoding untuk mengubah variabel kategori menjadi variabel numerik berdasarkan jumlah kemunculan (frekuensi) setiap kategori di dalam dataset. Metode ini sangat berguna untuk menangani variabel kategori dengan banyak nilai unik (high cardinality). 


Selanjutnya melakukan proses encoding pada fitur kategori yakni teknik *One-Hot Encoding*, fitur yang kita pilih adalah ```'Make', 'Vehicle Class', 'Transmission', 'Fuel Type'```






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Engine Size(L)</th>
      <th>Cylinders</th>
      <th>Fuel Consumption City (L/100 km)</th>
      <th>Fuel Consumption Hwy (L/100 km)</th>
      <th>Fuel Consumption Comb (L/100 km)</th>
      <th>Fuel Consumption Comb (mpg)</th>
      <th>CO2 Emissions(g/km)</th>
      <th>Make_ACURA</th>
      <th>Make_ALFA ROMEO</th>
      <th>Make_ASTON MARTIN</th>
      <th>Make_AUDI</th>
      <th>Make_BENTLEY</th>
      <th>Make_BMW</th>
      <th>Make_BUICK</th>
      <th>Make_CADILLAC</th>
      <th>Make_CHEVROLET</th>
      <th>Make_CHRYSLER</th>
      <th>Make_DODGE</th>
      <th>Make_FIAT</th>
      <th>Make_FORD</th>
      <th>Make_GENESIS</th>
      <th>Make_GMC</th>
      <th>Make_HONDA</th>
      <th>Make_HYUNDAI</th>
      <th>Make_INFINITI</th>
      <th>Make_JAGUAR</th>
      <th>Make_JEEP</th>
      <th>Make_KIA</th>
      <th>Make_LAMBORGHINI</th>
      <th>Make_LAND ROVER</th>
      <th>Make_LEXUS</th>
      <th>Make_LINCOLN</th>
      <th>Make_MASERATI</th>
      <th>Make_MAZDA</th>
      <th>Make_MERCEDES-BENZ</th>
      <th>Make_MINI</th>
      <th>Make_MITSUBISHI</th>
      <th>Make_NISSAN</th>
      <th>Make_PORSCHE</th>
      <th>Make_RAM</th>
      <th>Make_SCION</th>
      <th>Make_SMART</th>
      <th>Make_SUBARU</th>
      <th>Make_TOYOTA</th>
      <th>Make_VOLKSWAGEN</th>
      <th>Make_VOLVO</th>
      <th>Vehicle Class_COMPACT</th>
      <th>Vehicle Class_FULL-SIZE</th>
      <th>Vehicle Class_MID-SIZE</th>
      <th>Vehicle Class_MINICOMPACT</th>
      <th>Vehicle Class_MINIVAN</th>
      <th>Vehicle Class_PICKUP TRUCK - SMALL</th>
      <th>Vehicle Class_PICKUP TRUCK - STANDARD</th>
      <th>Vehicle Class_SPECIAL PURPOSE VEHICLE</th>
      <th>Vehicle Class_STATION WAGON - MID-SIZE</th>
      <th>Vehicle Class_STATION WAGON - SMALL</th>
      <th>Vehicle Class_SUBCOMPACT</th>
      <th>Vehicle Class_SUV - SMALL</th>
      <th>Vehicle Class_SUV - STANDARD</th>
      <th>Vehicle Class_TWO-SEATER</th>
      <th>Vehicle Class_VAN - CARGO</th>
      <th>Vehicle Class_VAN - PASSENGER</th>
      <th>Transmission_A10</th>
      <th>Transmission_A4</th>
      <th>Transmission_A5</th>
      <th>Transmission_A6</th>
      <th>Transmission_A7</th>
      <th>Transmission_A8</th>
      <th>Transmission_A9</th>
      <th>Transmission_AM5</th>
      <th>Transmission_AM6</th>
      <th>Transmission_AM7</th>
      <th>Transmission_AM8</th>
      <th>Transmission_AM9</th>
      <th>Transmission_AS10</th>
      <th>Transmission_AS4</th>
      <th>Transmission_AS5</th>
      <th>Transmission_AS6</th>
      <th>Transmission_AS7</th>
      <th>Transmission_AS8</th>
      <th>Transmission_AS9</th>
      <th>Transmission_AV</th>
      <th>Transmission_AV10</th>
      <th>Transmission_AV6</th>
      <th>Transmission_AV7</th>
      <th>Transmission_AV8</th>
      <th>Transmission_M5</th>
      <th>Transmission_M6</th>
      <th>Transmission_M7</th>
      <th>Fuel Type_D</th>
      <th>Fuel Type_E</th>
      <th>Fuel Type_N</th>
      <th>Fuel Type_X</th>
      <th>Fuel Type_Z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>2.0</td>
      <td>4</td>
      <td>9.9</td>
      <td>6.7</td>
      <td>8.5</td>
      <td>33</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2.4</td>
      <td>4</td>
      <td>11.2</td>
      <td>7.7</td>
      <td>9.6</td>
      <td>29</td>
      <td>221</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3.5</td>
      <td>6</td>
      <td>12.7</td>
      <td>9.1</td>
      <td>11.1</td>
      <td>25</td>
      <td>255</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3.5</td>
      <td>6</td>
      <td>12.1</td>
      <td>8.7</td>
      <td>10.6</td>
      <td>27</td>
      <td>244</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3.5</td>
      <td>6</td>
      <td>11.9</td>
      <td>7.7</td>
      <td>10.0</td>
      <td>28</td>
      <td>230</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Reduksi Dimensi dengan PCA

Selanjutnya yaitu melakukan reduksi dimensi pada fitur ```"Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)", "Fuel Consumption Comb (mpg)"``` karna fitur-fitur ini memiliki informasi yang sama yaitu konsumsi bahan bakar.






    array([0.979, 0.015, 0.005, 0.   ])



Berdasarkan hasil ini, kita akan mereduksi fitur (dimensi) dan hanya mempertahankan PC (komponen) pertama saja. PC pertama ini akan menjadi fitur dimensi, Beri nama fitur dengan ```'Fuel Consumptions'```.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Engine Size(L)</th>
      <th>Cylinders</th>
      <th>CO2 Emissions(g/km)</th>
      <th>Make_ACURA</th>
      <th>Make_ALFA ROMEO</th>
      <th>Make_ASTON MARTIN</th>
      <th>Make_AUDI</th>
      <th>Make_BENTLEY</th>
      <th>Make_BMW</th>
      <th>Make_BUICK</th>
      <th>Make_CADILLAC</th>
      <th>Make_CHEVROLET</th>
      <th>Make_CHRYSLER</th>
      <th>Make_DODGE</th>
      <th>Make_FIAT</th>
      <th>Make_FORD</th>
      <th>Make_GENESIS</th>
      <th>Make_GMC</th>
      <th>Make_HONDA</th>
      <th>Make_HYUNDAI</th>
      <th>Make_INFINITI</th>
      <th>Make_JAGUAR</th>
      <th>Make_JEEP</th>
      <th>Make_KIA</th>
      <th>Make_LAMBORGHINI</th>
      <th>Make_LAND ROVER</th>
      <th>Make_LEXUS</th>
      <th>Make_LINCOLN</th>
      <th>Make_MASERATI</th>
      <th>Make_MAZDA</th>
      <th>Make_MERCEDES-BENZ</th>
      <th>Make_MINI</th>
      <th>Make_MITSUBISHI</th>
      <th>Make_NISSAN</th>
      <th>Make_PORSCHE</th>
      <th>Make_RAM</th>
      <th>Make_SCION</th>
      <th>Make_SMART</th>
      <th>Make_SUBARU</th>
      <th>Make_TOYOTA</th>
      <th>Make_VOLKSWAGEN</th>
      <th>Make_VOLVO</th>
      <th>Vehicle Class_COMPACT</th>
      <th>Vehicle Class_FULL-SIZE</th>
      <th>Vehicle Class_MID-SIZE</th>
      <th>Vehicle Class_MINICOMPACT</th>
      <th>Vehicle Class_MINIVAN</th>
      <th>Vehicle Class_PICKUP TRUCK - SMALL</th>
      <th>Vehicle Class_PICKUP TRUCK - STANDARD</th>
      <th>Vehicle Class_SPECIAL PURPOSE VEHICLE</th>
      <th>Vehicle Class_STATION WAGON - MID-SIZE</th>
      <th>Vehicle Class_STATION WAGON - SMALL</th>
      <th>Vehicle Class_SUBCOMPACT</th>
      <th>Vehicle Class_SUV - SMALL</th>
      <th>Vehicle Class_SUV - STANDARD</th>
      <th>Vehicle Class_TWO-SEATER</th>
      <th>Vehicle Class_VAN - CARGO</th>
      <th>Vehicle Class_VAN - PASSENGER</th>
      <th>Transmission_A10</th>
      <th>Transmission_A4</th>
      <th>Transmission_A5</th>
      <th>Transmission_A6</th>
      <th>Transmission_A7</th>
      <th>Transmission_A8</th>
      <th>Transmission_A9</th>
      <th>Transmission_AM5</th>
      <th>Transmission_AM6</th>
      <th>Transmission_AM7</th>
      <th>Transmission_AM8</th>
      <th>Transmission_AM9</th>
      <th>Transmission_AS10</th>
      <th>Transmission_AS4</th>
      <th>Transmission_AS5</th>
      <th>Transmission_AS6</th>
      <th>Transmission_AS7</th>
      <th>Transmission_AS8</th>
      <th>Transmission_AS9</th>
      <th>Transmission_AV</th>
      <th>Transmission_AV10</th>
      <th>Transmission_AV6</th>
      <th>Transmission_AV7</th>
      <th>Transmission_AV8</th>
      <th>Transmission_M5</th>
      <th>Transmission_M6</th>
      <th>Transmission_M7</th>
      <th>Fuel Type_D</th>
      <th>Fuel Type_E</th>
      <th>Fuel Type_N</th>
      <th>Fuel Type_X</th>
      <th>Fuel Type_Z</th>
      <th>Fuel Consumptions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>2.0</td>
      <td>4</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6.555270</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2.4</td>
      <td>4</td>
      <td>221</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2.137705</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3.5</td>
      <td>6</td>
      <td>255</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-2.583639</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3.5</td>
      <td>6</td>
      <td>244</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.434697</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>3.5</td>
      <td>6</td>
      <td>230</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.909774</td>
    </tr>
  </tbody>
</table>
</div>



## Train Test Split

Selanjutnya lakukan pembagian data train dan test dengan perbandingan masing-masing 8:2 :


```python
X = df.drop(['CO2 Emissions(g/km)'], axis= 1)
y = df['CO2 Emissions(g/km)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```


```python
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')
```

    Total # of sample in whole dataset: 5816
    Total # of sample in train dataset: 4652
    Total # of sample in test dataset: 1164


## Standarisasi

Lakukan standarisasi pada kolom ```"Engine Size(L)", "Cylinders", dan "Fuel Consumptions"``` menggunakan *StandardScaler* dari pustaka Scikit-learn untuk mengubah data menjadi distribusi dengan rata-rata 0 dan standar deviasi 1. Proses ini membantu memastikan bahwa fitur-fitur tersebut memiliki skala yang sama, sehingga mendukung kinerja algoritma machine learning secara optimal. Standarisasi diterapkan hanya pada data latih untuk menghindari *data leakage*, dan transformasi yang sama digunakan pada data uji selama evaluasi.






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Engine Size(L)</th>
      <th>Cylinders</th>
      <th>Fuel Consumptions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2719</th>
      <td>-0.990016</td>
      <td>-0.907496</td>
      <td>1.174731</td>
    </tr>
    <tr>
      <th>5111</th>
      <td>-0.825232</td>
      <td>-0.907496</td>
      <td>1.172802</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>-0.825232</td>
      <td>-0.907496</td>
      <td>0.868077</td>
    </tr>
    <tr>
      <th>3013</th>
      <td>-0.825232</td>
      <td>-0.907496</td>
      <td>0.101519</td>
    </tr>
    <tr>
      <th>424</th>
      <td>0.410650</td>
      <td>0.414462</td>
      <td>-0.038515</td>
    </tr>
  </tbody>
</table>
</div>



proses standarisasi mengubah nilai rata-rata (mean) menjadi 0 dan nilai standar deviasi menjadi 1. Berikut untuk mengecek nilai mean dan standar deviasi pada setelah proses standarisasi.






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Engine Size(L)</th>
      <th>Cylinders</th>
      <th>Fuel Consumptions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4652.0000</td>
      <td>4652.0000</td>
      <td>4652.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.0001</td>
      <td>1.0001</td>
      <td>1.0001</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.7315</td>
      <td>-1.5685</td>
      <td>-2.3246</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.8252</td>
      <td>-0.9075</td>
      <td>-0.7432</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.0013</td>
      <td>0.4145</td>
      <td>-0.0458</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.4930</td>
      <td>0.4145</td>
      <td>0.7241</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.6352</td>
      <td>1.7364</td>
      <td>2.8319</td>
    </tr>
  </tbody>
</table>
</div>



# Model Development
___

Model yang digunakan untuk memprediksi tingkat emisi CO2 adalah ```KNN (K-Nearest Neighbors), Random Forest, dan Boosting```. **KNN** merupakan algoritma yang mengklasifikasikan atau memprediksi nilai berdasarkan kedekatannya dengan data lain dalam ruang fitur. **Random Forest** adalah algoritma ensemble yang membangun banyak pohon keputusan dan menggabungkan hasil prediksi mereka untuk meningkatkan akurasi dan mengurangi overfitting. Sedangkan **Boosting**, khususnya AdaBoost, adalah teknik ensemble yang menggabungkan model-model lemah secara berturut-turut, di mana model berikutnya berfokus pada kesalahan yang dibuat oleh model sebelumnya, untuk meningkatkan performa prediksi secara keseluruhan. Masing-masing model ini memiliki pendekatan yang berbeda dalam memproses data dan mengoptimalkan prediksi, yang memungkinkan untuk membandingkan kekuatan prediktif mereka dalam memprediksi emisi CO2.


```python
# Menyiapkan dataframe untuk analisis
model = pd.DataFrame(index=['train_mse', 'test_mse'], columns=['KNN', 'RandomForest', 'Boosting'])
```

## Model Development dengan KNN

Algoritma *K-Nearest Neighbors* (KNN) adalah metode non-parametrik yang digunakan untuk regresi atau klasifikasi. Dalam regresi, KNN memprediksi nilai target untuk sampel baru berdasarkan rata-rata nilai target dari **k tetangga terdekat** dalam ruang fitur.

Pada kode ini, KNN diterapkan menggunakan `KNeighborsRegressor` dari Scikit-learn, dengan parameter `n_neighbors=10`. Parameter ini menentukan bahwa prediksi untuk setiap data akan didasarkan pada rata-rata nilai target dari 10 tetangga terdekat di data latih (`X_train`). Tetangga terdekat dihitung berdasarkan metrik jarak, biasanya jarak Euclidean secara default.

Model ini kemudian dilatih menggunakan metode `.fit()` pada data latih (`X_train` dan `y_train`). Setelah pelatihan, nilai Mean Squared Error (MSE) dihitung untuk memeriksa seberapa baik model memprediksi data latih dengan membandingkan hasil prediksi `knn.predict(X_train)` terhadap nilai sebenarnya `y_train`. Metrik MSE ini kemudian disimpan ke dalam tabel `model` untuk keperluan analisis lebih lanjut.

KNN cenderung sensitif terhadap pilihan nilai `n_neighbors`:
- **Nilai kecil** (misalnya 1) dapat menyebabkan model overfit.
- **Nilai besar** dapat menyebabkan model underfit karena menghaluskan prediksi terlalu banyak.

Pemilihan parameter ini sebaiknya disesuaikan melalui *cross-validation*.


```python
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

model.loc['train_mse', 'knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
```

## Model Development dengan Random Forest

**Random Forest** adalah algoritma ensemble berbasis pohon keputusan yang digunakan untuk tugas regresi atau klasifikasi. Algoritma ini membangun beberapa pohon keputusan selama pelatihan dan menggabungkan prediksi mereka untuk meningkatkan akurasi dan mengurangi risiko overfitting.

Pada kode ini, Random Forest diterapkan menggunakan `RandomForestRegressor` dari Scikit-learn. Parameter yang digunakan adalah sebagai berikut:  
- **`n_estimators=50`**: Membuat 50 pohon keputusan sebagai bagian dari ensemble.  
- **`max_depth=20`**: Membatasi kedalaman maksimum setiap pohon untuk menghindari overfitting.  
- **`random_state=321`**: Menetapkan *seed* untuk hasil yang dapat direproduksi.  
- **`n_jobs=-1`**: Menggunakan semua inti prosesor untuk mempercepat pelatihan model.  


```python
RF = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=321, n_jobs=-1)
RF.fit(X_train, y_train)

model.loc['train_mse', 'RandomForest'] = mean_squared_error(y_pred = RF.predict(X_train), y_true=y_train)
```

## Model Development dengan Boosting

**Boosting** adalah metode ensemble yang secara iteratif meningkatkan performa model dengan menggabungkan beberapa model lemah (weak learners) untuk membentuk model yang lebih kuat. Pada kode ini, Boosting diterapkan menggunakan algoritma **AdaBoostRegressor** dari Scikit-learn.

### Penjelasan Parameter
- **`learning_rate=0.05`**: Mengontrol kontribusi setiap model lemah terhadap hasil akhir. Nilai yang lebih kecil dapat meningkatkan stabilitas dan mengurangi risiko overfitting.  
- **`random_state=321`**: Menetapkan *seed* untuk hasil yang dapat direproduksi.  


```python
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=321)
boosting.fit(X_train, y_train)

model.loc['train_mse', 'Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```

# Model Evaluation
___

Evaluasi model adalah proses untuk mengukur performa model prediktif dengan menggunakan data yang tidak dilibatkan dalam proses pelatihan. Salah satu metrik evaluasi yang umum digunakan untuk masalah regresi adalah **Mean Squared Error (MSE)**.



### **Mean Squared Error (MSE)**
MSE dihitung sebagai rata-rata kuadrat selisih antara nilai sebenarnya dan nilai prediksi. Rumusnya adalah:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

$$
\begin{aligned}
y_i & : \text{Nilai aktual (observasi sebenarnya)} \\
\hat{y}_i & : \text{Nilai prediksi model} \\
n & : \text{Jumlah data (observasi)}
\end{aligned}
$$

Nilai MSE yang lebih kecil menunjukkan model memiliki kemampuan prediksi yang lebih baik, karena kesalahan prediksi lebih kecil.

## Persiapan Data Test

Pastikan kolom numerik di X_test diubah menjadi tipe float agar kompatibel dengan proses transformasi skala. Kolom numerik di X_test diubah menjadi distribusi dengan rata-rata = 0 dan standar deviasi = 1 menggunakan scaler yang sudah dilatih sebelumnya pada data latih (scaler.transform). Hasil transformasi kemudian disimpan kembali ke X_test untuk menjaga konsistensi data.


```python
# Pastikan kolom numerik di X_test diubah menjadi float terlebih dahulu
X_test[numerical_features] = X_test[numerical_features].astype(float)

# Scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
numerical_features = list(numerical_features)

# Lakukan scaling pada data numerik
scaled_values = scaler.transform(X_test[numerical_features])

# Buat DataFrame dari hasil transformasi dengan tipe data konsisten
scaled_df = pd.DataFrame(scaled_values, columns=numerical_features, index=X_test.index)

# Gantikan kolom lama dengan kolom baru yang sudah diskalakan
X_test.loc[:, numerical_features] = scaled_df

```

Dengan menggunakan dictionary berisi model yang telah dilatih, MSE dihitung untuk data latih (X_train, y_train) dan data uji (X_test, y_test) menggunakan fungsi mean_squared_error. Nilai MSE disimpan dalam DataFrame mse, dan dibagi dengan 1000 untuk skala yang lebih mudah dibaca.


```python
# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN':knn, 'RF':RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

mse
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KNN</th>
      <td>0.133788</td>
      <td>0.158733</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>0.002169</td>
      <td>0.007982</td>
    </tr>
    <tr>
      <th>Boosting</th>
      <td>0.135352</td>
      <td>0.128702</td>
    </tr>
  </tbody>
</table>
</div>






    
![png](gambar_files/gambar_104_0.png)
    


Hasil MSE diperoleh bahwa algoritma Random Forest memiliki MSE yang sangat rendah, meskipun pada data test lebih tinggi namun masih bisa ditolerir. Pada algoritma Boosting diperoleh MSE pada data test lebih rendah dari data train, jika dibanding dengan Random Forest MSE Boosting lebih tinggi namun masih kategori nilai yang rendah, bahkan bisa dibilang Boosting lebih stabil karna menunjukkan MSE pada test lebih rendah. untuk algoritma KNN, MSE pada test masih sangat tinggi dibandingkan pada data train






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_true</th>
      <th>prediksi_KNN</th>
      <th>prediksi_RF</th>
      <th>prediksi_Boosting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5063</th>
      <td>280</td>
      <td>282.8</td>
      <td>279.855</td>
      <td>279.654</td>
    </tr>
    <tr>
      <th>5637</th>
      <td>175</td>
      <td>174.8</td>
      <td>176.155</td>
      <td>170.540</td>
    </tr>
    <tr>
      <th>6979</th>
      <td>236</td>
      <td>240.7</td>
      <td>233.777</td>
      <td>239.856</td>
    </tr>
  </tbody>
</table>
</div>



# Kesimpulan
Untuk performa keseluruhan, Random Forest tetap menjadi model terbaik berdasarkan MSE yang rendah, dan hasil prediksi yang paling akurat jika dibandingkan dengan kedua model yang lain, kemudian model Boosting bisa menjadi pilihan selanjutnya karna lebih stabil, jika dibandingkan dengan KNN yang nilai MSE pada data train cukup tinggi.

**Distribusi Variabel Kategori:**
- **Make**: Ford, BMW, dan Chevrolet paling sering digunakan, sementara Lamborghini paling rendah.
- **Vehicle Class**: SUV-Small, Mid-Size, dan Compact adalah yang paling umum, sedangkan Van-Cargo paling sedikit digunakan.
- **Transmission**: AS6, AS8, M6, dan A6 adalah transmisi yang paling umum, sementara AV10, AM5, AS4, dan AM9 paling jarang.
- **Fuel Type**: Regular (X) dan Premium (Z) adalah bahan bakar yang paling sering digunakan, sedangkan E85 (E) dan Diesel (D) digunakan dengan rentang 147-223, sementara Natural Gas (N) hanya ada satu entri.

**Distribusi Fitur Numerik:**
- **Engine Size (L)**: Ukuran mesin antara 2-4 liter adalah yang paling umum.
- **Cylinders**: Mobil dengan 4, 6, dan 8 silinder adalah yang paling banyak digunakan.
- **Fuel Consumption (City)**: Konsumsi bahan bakar paling sering di kisaran 10-13L per 100 km.
- **Fuel Consumption (Hwy)**: Konsumsi bahan bakar paling banyak di kisaran 7-9L per 100 km.
- **Fuel Consumption (Comb)**: Paling banyak di kisaran 8-12L per 100 km.
- **Fuel Consumption (mpg)**: Kisaran 20-35 miles per gallon paling banyak digunakan.
- **CO2 Emissions**: Emisi terbanyak berada di kisaran 200-250 g/km.

**Distribusi Kategori terhadap CO2 Emissions:**
- **Make**: Merek seperti ```BUGATTI, ROLLS-ROYCE, SRT, LAMBORGHINI, ASTON MARTIN, BENTLEY, dan MASERATI``` memiliki emisi tertinggi, tetapi penggunaannya rendah.
- **Vehicle Class**: Van-Cargo dan Van-Passenger menyumbang emisi tertinggi meskipun jarang digunakan. SUV-Small, Mid-Size, dan Compact menghasilkan emisi antara 200-250 g/km.
- **Transmission**: A10, A7, dan A5 menyumbang emisi tertinggi meskipun jarang digunakan. AS6, AS8, dan M6, yang lebih sering digunakan, menyumbang emisi dalam kisaran 200-250 g/km.
- **Fuel Type**: Semua jenis bahan bakar menyumbang emisi tinggi, meskipun tipe N (Natural Gas) hanya memiliki satu data, sehingga tidak bisa disimpulkan.

**Relasi antara Fitur Numerik dan CO2 Emissions:**
- Terdapat relasi positif antara sebagian besar fitur numerik (seperti Engine Size, Cylinders, dan Fuel Consumption) dengan CO2 emissions, yang berarti semakin besar ukuran mesin, jumlah silinder, atau konsumsi bahan bakar, semakin tinggi emisi yang dihasilkan.
- **Fuel Consumption (mpg)** menunjukkan relasi negatif, karena semakin tinggi efisiensi bahan bakar, semakin rendah emisi CO2 yang dihasilkan.


# Referensi

> 1. Anggraeni, D. N. (2017). Kajian Emisi CO2 Dari Kendaraan Bermotor Di Kampus I Universitas Brawijaya Dan Kampus I Universitas Negeri Malang. Retrieved from https://api.semanticscholar.org/CorpusID:127584962

> 2. Sudarti, S., Yushardi, Y., & Kasanah, N. (2022). Analisis Potensi Emisi CO2 Oleh Berbagai Jenis Kendaraan Bermotor di Jalan Raya Kemantren Kabupaten Sidoarjo. *Jurnal Sumberdaya Alam dan Lingkungan*. Retrieved from https://api.semanticscholar.org/CorpusID:252253520

> 3. Kusumawardani, D., & Navastara, A. M. (2018). Analisis Besaran Emisi Gas CO2 Kendaraan Bermotor Pada Kawasan Industri SIER Surabaya. *Jurnal Teknik ITS*, 6, 399-402. Retrieved from https://api.semanticscholar.org/CorpusID:115275286

> 4. Kusumawardani, D. (2017). Arahan Penyediaan Ruang Terbuka Hijau Dalam Menyerap Emisi Gas Co2 Kendaraan Bermotor Pada Kawasan Industri Sier, Surabaya. Retrieved from https://api.semanticscholar.org/CorpusID:169900032

> 5. Amelia, C. R., Samadikun, B. P., & Huboyo, H. S. (2017). Analisis Shifting Penggunaan Moda Kendaraan Bermotor Ke Kereta Api Terhadap Penurunan Emisi Gas Rumah Kaca (CO2, CH4, dan N2O) Studi Kasus: Daerah Operasional VIII Surabaya. Retrieved from https://api.semanticscholar.org/CorpusID:114581634

> 6. Paradizsa, I. (2023). Analisis Kebijakan Pengendalian Polusi melalui Uji Emisi Kendaraan Bermotor Berbahan Bakar Minyak (BBM) di Wilayah DKI Jakarta. *Jurnal Enviscience*. Retrieved from https://api.semanticscholar.org/CorpusID:268714970

> 7. Buanawati, T. T., Huboyo, H. S., & Samadikun, B. P. (2017). Estimasi Emisi Pencemar Udara Konvensional (SOx, NOx, CO, dan PM) Kendaraan Pribadi Berdasarkan Metode International Vehicle Emission (IVE) di Beberapa Ruas Jalan Kota Semarang. Retrieved from https://api.semanticscholar.org/CorpusID:117120421

