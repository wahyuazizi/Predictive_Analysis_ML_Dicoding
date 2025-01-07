```python
# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

```


    VBox(children=(HTML(value='<center> <img\nsrc=https://www.kaggle.com/static/images/site-logo.png\nalt=\'Kaggle…


    Kaggle credentials set.
    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.5), please consider upgrading to the latest version (0.3.6).
    Kaggle credentials successfully validated.



```python
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

brsahan_vehicle_co2_emissions_dataset_path = kagglehub.dataset_download('brsahan/vehicle-co2-emissions-dataset')
wahyuazizi_kagglesjson_path = kagglehub.dataset_download('wahyuazizi/kagglesjson')

print('Data source import complete.')

```

    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.5), please consider upgrading to the latest version (0.3.6).
    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.5), please consider upgrading to the latest version (0.3.6).
    Downloading from https://www.kaggle.com/api/v1/datasets/download/wahyuazizi/kagglesjson?dataset_version_number=1...


    100%|██████████| 228/228 [00:00<00:00, 565kB/s]

    Extracting files...
    Data source import complete.


    


# Laporan Proyek Machine Learning - Wahyu Azizi

## Domain Proyek

Peningkatan emisi karbon dioksida (CO2) dari kendaraan bermotor menjadi salah satu penyumbang utama perubahan iklim global. Hal ini disebabkan oleh pertumbuhan jumlah kendaraan yang pesat, terutama di negara-negara berkembang seperti Indonesia. Kendaraan bermotor, terutama yang menggunakan bahan bakar fosil, menghasilkan emisi CO2 melalui proses pembakaran. Penelitian menunjukkan bahwa sektor transportasi berkontribusi signifikan terhadap total emisi gas rumah kaca. Di Indonesia, emisi dari transportasi diperkirakan mencapai 85% dari total pencemaran udara, dengan kendaraan pribadi dan sepeda motor sebagai penyumbang terbesar [[1](https://www.semanticscholar.org/paper/Kajian-Emisi-CO2-Dari-Kendaraan-Bermotor-Di-Kampus-Anggraeni/214a377555002c31829c59dbf1dea7cfb396ef8d)].  
Sebuah studi di Jalan Raya Kemantren, Sidoarjo, menunjukkan bahwa potensi emisi CO2 dari sepeda motor mencapai 67.568,26 g dalam 30 menit per kilometer, sedangkan mobil menghasilkan sekitar 63.335,30 g pada periode yang sama[[2](https://www.semanticscholar.org/paper/Analisis-Potensi-Emisi-CO2-Oleh-Berbagai-Jenis-di-Sudarti-Yushardi/81d79cda27154bdf4c54548423a878659b872df5)]. Selain itu, kawasan industri seperti SIER di Surabaya juga mengalami peningkatan emisi CO2 akibat tingginya kepadatan kendaraan dan kemacetan[[3](https://www.semanticscholar.org/paper/Analisis-Besaran-Emisi-Gas-CO2-Kendaraan-Bermotor-Kusumawardani-Navastara/54310e88f91c19e3519403d20b70adf299d55a3b)][[4](https://www.semanticscholar.org/paper/Arahan-Penyediaan-Ruang-Terbuka-Hijau-Dalam-Emisi-Kusumawardani/52438bef01b4397f8f43cc90d4a3a8dafc5edea6)]. Peningkatan emisi CO2 berkontribusi pada efek rumah kaca, yang berujung pada pemanasan global. Kenaikan suhu akibat peningkatan konsentrasi gas rumah kaca dapat menyebabkan perubahan iklim yang ekstrem, termasuk cuaca yang tidak menentu dan bencana alam[[5](https://www.semanticscholar.org/paper/ANALISIS-SHIFTING-PENGGUNAAN-MODA-KENDARAAN-KE-API-Amelia-Samadikun/9b4974abd17e7a013948ebb8f2ca4cf11b924e60)]. Penelitian juga menunjukkan bahwa daerah dengan kepadatan transportasi tinggi mengalami penurunan kualitas udara dan peningkatan suhu[[3](https://www.semanticscholar.org/paper/Analisis-Besaran-Emisi-Gas-CO2-Kendaraan-Bermotor-Kusumawardani-Navastara/54310e88f91c19e3519403d20b70adf299d55a3b)].  
Kualitas udara yang buruk diakibatkan oleh peningkatan jumlah kendaraan bermotor yang menghasilkan emisi gas berbahaya seperti karbon dioksida (CO2), nitrogen oksida (NOx), dan partikel halus (PM). Di DKI Jakarta, misalnya, kebijakan uji emisi diharapkan dapat mengurangi polusi, namun tantangan dalam pengawasan dan pertumbuhan jumlah kendaraan tetap menjadi masalah besar[[6](https://www.semanticscholar.org/paper/Analisis-Kebijakan-Pengendalian-Polusi-melalui-Uji-Paradizsa/72478900647357dab65f36a62ac558041e1dff34)]. Penelitian menunjukkan bahwa emisi dari kendaraan pribadi berkontribusi signifikan terhadap pencemaran udara di daerah dengan kepadatan lalu lintas tinggi, seperti Kota Semarang[[7](https://www.semanticscholar.org/paper/ESTIMASI-EMISI-PENCEMAR-UDARA-KONVENSIONAL-(SOx%2C-DI-Buanawati-Huboyo/1aa8e1dc1be0e8e43c33b4a19a806625c8f01d15)].  
Data science memainkan peran penting dalam memahami dan menganalisis pola emisi dari kendaraan bermotor. Dengan memanfaatkan teknik analisis data, machine learning, dan pemodelan statistik, para peneliti dan pembuat kebijakan dapat memperoleh wawasan yang lebih baik mengenai sumber emisi, tren, dan dampaknya terhadap lingkungan.

# Business Understanding

## Problem Statement

Peningkatan emisi karbon dioksida (CO2) dari kendaraan bermotor menjadi salah satu penyumbang utama perubahan iklim global. Di Indonesia, sektor transportasi menyumbang hingga 85% dari total pencemaran udara, dengan kendaraan pribadi dan sepeda motor sebagai kontributor terbesar. Emisi ini terutama dipengaruhi oleh berbagai karakteristik kendaraan, termasuk ukuran mesin, jenis bahan bakar, dan efisiensi bahan bakar. Namun, tantangan muncul dari banyaknya faktor yang memengaruhi emisi, sehingga sulit untuk memahami pola emisi tanpa analisis yang mendalam.

Predictive analytics memberikan peluang untuk memanfaatkan data kendaraan dan mengembangkan model prediktif guna memahami dan memitigasi emisi CO2. Dengan memanfaatkan data seperti konsumsi bahan bakar, jenis transmisi, dan kapasitas mesin, model dapat membantu memprediksi emisi CO2 dengan akurasi tinggi.

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

# Import Library

Import library yang diperlukan untuk predictive analytic


```python
!pip install --upgrade seaborn
```

    Requirement already satisfied: seaborn in /usr/local/lib/python3.10/dist-packages (0.13.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in /usr/local/lib/python3.10/dist-packages (from seaborn) (1.26.4)
    Requirement already satisfied: pandas>=1.2 in /usr/local/lib/python3.10/dist-packages (from seaborn) (2.2.2)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /usr/local/lib/python3.10/dist-packages (from seaborn) (3.8.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.55.3)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.7)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.2)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (11.0.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.0)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.2->seaborn) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.2->seaborn) (2024.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.17.0)



```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import seaborn as sns
import kagglehub
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from wordcloud import WordCloud
```

# Data Understanding

Dataset ini berisi informasi tentang spesifikasi kendaraan, konsumsi bahan bakar, dan emisi CO2 yang dikumpulkan untuk menganalisis dampak lingkungan dari kendaraan serta memprediksi emisi CO2 menggunakan model regresi. Dataset ini dirancang untuk mendukung pendekatan **Simple Linear Regression (SLR)** dan **Multiple Linear Regression (MLR)** dalam proyek machine learning.

Tautannya dapat diakses di sini:  
[Vehicle CO2 Emissions Dataset](https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset)

## Data Loading

Sebelum melakukan eksplorasi data, terlebih dahulu kita load data, seperti berikut:


```python
# Download
path = kagglehub.dataset_download("brsahan/vehicle-co2-emissions-dataset")

print("path dataset: ", path)
```

    Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.5), please consider upgrading to the latest version (0.3.6).
    path dataset:  /root/.cache/kagglehub/datasets/brsahan/vehicle-co2-emissions-dataset/versions/1



```python
# Cek list file dalam path
files = os.listdir(path)
print("daftar file dalam path:")
for file in files:
    print(file)
```

    daftar file dalam path:
    co2.csv



```python
# Load dataset
df = pd.read_csv(f"{path}/{file}", low_memory=False)
print(df.shape)

pd.set_option('display.max_columns', None)
df.head()
```

    (7385, 12)






  <div id="df-d11ffa9e-68be-43b2-9122-f7698354d202" class="colab-df-container">
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d11ffa9e-68be-43b2-9122-f7698354d202')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d11ffa9e-68be-43b2-9122-f7698354d202 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d11ffa9e-68be-43b2-9122-f7698354d202');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-87ed0ee1-4065-4e1e-9d3f-0d1f55bcc917">
  <button class="colab-df-quickchart" onclick="quickchart('df-87ed0ee1-4065-4e1e-9d3f-0d1f55bcc917')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-87ed0ee1-4065-4e1e-9d3f-0d1f55bcc917 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
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



```python
df.info()
```

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

#### Menangani Missing Value & Duplicate Data


```python
pd.DataFrame({'Daftar Missing Value:':df.isnull().sum()})
```





  <div id="df-0bcaf1db-f0d2-4357-ab45-e6d56205b01d" class="colab-df-container">
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0bcaf1db-f0d2-4357-ab45-e6d56205b01d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0bcaf1db-f0d2-4357-ab45-e6d56205b01d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0bcaf1db-f0d2-4357-ab45-e6d56205b01d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e25437e4-5fcc-4fa1-a8e6-ea101ad68b82">
  <button class="colab-df-quickchart" onclick="quickchart('df-e25437e4-5fcc-4fa1-a8e6-ea101ad68b82')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e25437e4-5fcc-4fa1-a8e6-ea101ad68b82 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Menampilkan baris duplikat
duplicates = df[df.duplicated()]

# Jumlah baris duplikat
print("Jumlah duplikat:", df.duplicated().sum())
```

    Jumlah duplikat: 1103



```python
df = df.drop_duplicates()
# Memeriksa ulang jumlah duplikat
print("Jumlah duplikat setelah pembersihan:", df.duplicated().sum())
```

    Jumlah duplikat setelah pembersihan: 0


Berdasarkan informasi diatas, dataset tersebut tidak memiliki missing value sama sekali, akan tetapi memiliki duplicate value yang lumayan banyak yaitu sebanyak ```1103``` data yang terduplikasi, sehingga harus dihilangkan.

#### Menangani Outliers


```python
df.describe()
```





  <div id="df-0ceda942-f28c-48ca-a23a-96242e671d34" class="colab-df-container">
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
      <td>6282.000000</td>
      <td>6282.000000</td>
      <td>6282.000000</td>
      <td>6282.000000</td>
      <td>6282.000000</td>
      <td>6282.000000</td>
      <td>6282.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.161812</td>
      <td>5.618911</td>
      <td>12.610220</td>
      <td>9.070583</td>
      <td>11.017876</td>
      <td>27.411016</td>
      <td>251.157752</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.365201</td>
      <td>1.846250</td>
      <td>3.553066</td>
      <td>2.278884</td>
      <td>2.946876</td>
      <td>7.245318</td>
      <td>59.290426</td>
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
      <td>14.700000</td>
      <td>10.300000</td>
      <td>12.700000</td>
      <td>32.000000</td>
      <td>289.000000</td>
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0ceda942-f28c-48ca-a23a-96242e671d34')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0ceda942-f28c-48ca-a23a-96242e671d34 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0ceda942-f28c-48ca-a23a-96242e671d34');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-48f0413c-f6b1-402f-848b-86c69b66be48">
  <button class="colab-df-quickchart" onclick="quickchart('df-48f0413c-f6b1-402f-848b-86c69b66be48')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-48f0413c-f6b1-402f-848b-86c69b66be48 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




dari ringkasan statistik yang diperoleh:  

---

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

---

##### **Catatan Tambahan:**
- **Outlier**: Nilai yang jauh di luar rentang normal, seperti konsumsi bahan bakar kombinasi yang mencapai **26.10 L/100 km** atau emisi CO2 sebesar **522 g/km**, dan variabel mungkin memerlukan perhatian lebih karena berpotensi menjadi *outlier*.
- **Variabilitas**: Fitur seperti konsumsi bahan bakar dan emisi CO2 menunjukkan tingkat variabilitas yang cukup tinggi, seperti yang ditunjukkan oleh standar deviasi yang besar.


```python
numerical_features = df.select_dtypes(include=np.number).columns

n_cols = 2
n_rows = (len(numerical_features) + n_cols -1) // n_cols
```


```python
# setup figure dengan subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 1.5 * n_rows))

axes = axes.flatten()
for i in range(len(numerical_features)):
    sns.boxplot(ax=axes[i], x=df[numerical_features[i]])
    axes[i].set_title(numerical_features[i])

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_38_0.png)
    


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



```python
numerical_features = df.select_dtypes(include=['number'])

Q1 = numerical_features.quantile(0.25)
Q3 = numerical_features.quantile(0.75)
IQR = Q3 - Q1

# Menyaring outlier berdasarkan IQR
df = df[~((numerical_features < (Q1 - 1.5 * IQR)) | (numerical_features > (Q3 + 1.5 * IQR))).any(axis=1)]

```


```python
df.shape
```




    (5816, 12)




```python
df.describe()
```





  <div id="df-ec0cad18-1f53-4134-88ea-aa4756eaac24" class="colab-df-container">
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ec0cad18-1f53-4134-88ea-aa4756eaac24')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ec0cad18-1f53-4134-88ea-aa4756eaac24 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ec0cad18-1f53-4134-88ea-aa4756eaac24');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e7ddf807-e0ed-429a-a5fa-5ee31c47d875">
  <button class="colab-df-quickchart" onclick="quickchart('df-e7ddf807-e0ed-429a-a5fa-5ee31c47d875')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e7ddf807-e0ed-429a-a5fa-5ee31c47d875 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




### EDA - Univariate Analysis

#### Word Cloud


```python
# Gabungkan semua nilai dalam kolom 'Model' menjadi satu string
models_text = " ".join(df['Model'].dropna())

# Buat Word Cloud
wordcloud = WordCloud(
    width=800, height=400, background_color='white',
    colormap='viridis'
).generate(models_text)

# Tampilkan Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_45_0.png)
    


Variabel Model memiliki value sangat beragam sehingga visualisasi word cloud adalah yang cocok. Dimana Model sepet ```AWD```, ```COOPER```, ```TYPE```, ```FFV```, ```ROVER``` lebih banyak muncul dalam data

#### Categorical Features


```python
# Seleksi fitur kategorikal, dengan mengabaikan kolom 'Model' jika ada
categorical_features = df.select_dtypes(include=['object']).drop(columns=['Model'], errors='ignore')

# Tentukan jumlah kolom dan baris untuk plot
n_cols = 2
n_rows = (len(categorical_features.columns) + n_cols - 1) // n_cols  # Menghitung jumlah baris

# Membuat subplots dengan dua kolom
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))

# Loop untuk setiap fitur kategorikal
for i, cat_feature in enumerate(categorical_features.columns):
    # Hitung distribusi frekuensi untuk masing-masing fitur kategorikal
    count = df[cat_feature].value_counts()

    # Menentukan subplot saat ini
    ax = axes.flatten()[i]

    # Membuat horizontal bar plot dengan colormap 'viridis' pada subplot tertentu
    sns.barplot(x=count.values, y=count.index, hue=count, palette='viridis', ax=ax)

    # Menambahkan label dan judul
    ax.set_title(f'{cat_feature} Distribution')
    ax.set_xlabel('Count')
    ax.set_ylabel(cat_feature)

# Menyesuaikan layout agar plot tidak saling tumpang tindih
plt.tight_layout()
plt.show()
```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_48_0.png)
    


Berikutnya, distribusi tiap variabel kategori.  
- Distribusi ```Make```  ada vehicle dari ```FORD, BMW, CHEVROLET``` merupakan tiga teratas yang paling banyak digunakan dan ```LAMBORGHINI``` yang paling rendah.
- Pada variabel ```Vehicle Class``` ada class```SUV-SMALL, MID-SIZE, COMPACT``` yang merupakan top three yang paling banyak digunakan dan class ```VAN-CARGO``` adalah yang paling sedikit.
- Untuk variabel  ```Transmission``` terdapat ```AS6, AS8, M6, A6``` yang merupakan empat teratas untuk transmisi yang paling banyak digunakan dan transmisi ```AV10, AM5, AS4, AM9``` merupakan yang paling rendah.
- Dan pada variabel ```Fuel Type```, Jenis ```X (Regular) dan Z (Premium)``` adalah tipe bahan bakar yang paling umum digunakan, type ```E (E85 Campuran Bensin dan Etanol) dan D (Diesel)``` dengan rentang 147-223, sedangkan ```N (Natural Gas)``` hanya satu dalam data.

#### Numerical Features


```python
df.hist(bins=50, figsize=(20,15))
plt.show()
```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_51_0.png)
    


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


```python
df.drop("Model", axis=1, inplace=True)
```


```python
y = ['CO2 Emissions(g/km)']
x = ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type']

for f in y:
    for col in x:
        # Buat catplot
        g = sns.catplot(
            data=df, x=col, y=f, kind='bar', dodge=False, height=4, aspect=3, hue=col
        )
        # Tambahkan judul
        g.fig.suptitle(f"Rata-rata '{f}' relatif terhadap - {col}", y=1.02)
        # Rotasi sumbu x dengan benar
        for label in g.ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

        # Tampilkan plot
        plt.show()
```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_56_0.png)
    



    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_56_1.png)
    



    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_56_2.png)
    



    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_56_3.png)
    


**Distribusi fitur kategori terhadap CO2 Emissions**  

- Fitur ```Make``` dengan penyumbang emisi terbanyak adalah ```LAMBORGHINI, ASTON MARTIN, BENTLEY, dan MASERATI``` akan tetapi penggunaannya sangat-sangat rendah, bisa dilihat pada barplot horizontal di fase *Univariate Analysis*, dan rata-rata menyumbang emisi rentang 200-250 g/km
- Penyumbang emisi terbanyak pada fitur ```Vehicle Class``` adalah ```VAN-CARGO, dan VAN-PASSENGER``` tapi merupakan class yang paling sedikit digunakan. sedangkan untuk class yang paling banyak digunakan yakni ```SUV-SMALL, MID-SIZE, dan COMPACT``` menyumbang emisi diantara rentang 200-250 g/km
- Selanjutnya jenis ```Transmission``` yang paling banyak menyumbang emisi CO2 adalah jenis ```A10, A7, dan A5``` tapi merupakan jenis tranmisi yang tergolong rendah untuk penggunaannya. Untuk Transmisi yang paling sering digunakan yakni ```AS6, AS8, dan M6``` menyumbang emisi dengan rentang 200-250 g/km
- Rata-rata semua ```Fuel Type``` menyumbang emisi yang tinggi, akan tetapi type ```N``` belum bisa disimpulkan karna data yang ada hanya satu dalam dataset.

#### Numerical Features


```python
sns.pairplot(df, diag_kind='kde', hue="Fuel Type")
plt.show()
```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_59_0.png)
    


Relasi antara fitur numerik dengan fitur target yakni ```CO2 Emissions``` positif kecuali fitur ```Fuel Consumption(mpg)```, ini karna metode perhitungan miles per galon menghitung total jarak yang ditempuh per galonnya, semakin tinggi nilai tempuhnya maka semakin sedikit emisi yang dikeluarkan. nilai relasi bisa dilihat dalam hetmap berikut.


```python
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
plt.figure(figsize=(8,6))

corr_matrix = df[numerical_features].corr().round(2)
#gunakan annot=True untuk menampilkan nilai dalam kotak
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', linewidth=0.5)
plt.title("Correlation Matrix Untuk Numerical Features", size=20)
plt.show()
```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_61_0.png)
    


# Data Preparation

## Encoding Fitur Kategori

Selanjutnya melakukan proses encoding pada fitur kategori yakni teknik *One-Hot Encoding*, fitur yang kita pilih adalah ```'Make', 'Vehicle Class', 'Transmission', 'Fuel Type'```


```python
fitur = ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type']
for i in fitur:
    # Buat one-hot encoding dan ubah tipe menjadi integer eksplisit
    dummies = pd.get_dummies(df[i], prefix=f'{i}').astype(int)
    df = pd.concat([df, dummies], axis=1)
    df.drop(i, axis=1, inplace=True)

df.head()
```





  <div id="df-2402371e-8575-420e-951e-fbf138a34bf4" class="colab-df-container">
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2402371e-8575-420e-951e-fbf138a34bf4')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-2402371e-8575-420e-951e-fbf138a34bf4 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2402371e-8575-420e-951e-fbf138a34bf4');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-779d8728-0f94-440b-8397-e696db447580">
  <button class="colab-df-quickchart" onclick="quickchart('df-779d8728-0f94-440b-8397-e696db447580')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-779d8728-0f94-440b-8397-e696db447580 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




## Reduksi Dimensi dengan PCA

Selanjutnya yaitu melakukan reduksi dimensi pada fitur ```"Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)", "Fuel Consumption Comb (mpg)"``` karna fitur-fitur ini memiliki informasi yang sama yaitu konsumsi bahan bakar.


```python
fuel_features = ["Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)", "Fuel Consumption Comb (mpg)"]
pca = PCA(n_components=4, random_state=123)
pca.fit(df[fuel_features])
princ_comp = pca.transform(df[fuel_features])
```


```python
pca.explained_variance_ratio_.round(3)
```




    array([0.979, 0.015, 0.005, 0.   ])



Berdasarkan hasil ini, kita akan mereduksi fitur (dimensi) dan hanya mempertahankan PC (komponen) pertama saja. PC pertama ini akan menjadi fitur dimensi, Beri nama fitur dengan ```'Fuel Consumptions'```.


```python
pca = PCA(n_components=1, random_state=123)
pca.fit(df[fuel_features])
df['Fuel Consumptions'] = pca.transform(df.loc[:, (fuel_features)]).flatten()
df.drop(fuel_features, axis=1, inplace=True)
```


```python
df.head()
```





  <div id="df-d3b1b425-02ff-4358-983d-aae27ad8200d" class="colab-df-container">
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d3b1b425-02ff-4358-983d-aae27ad8200d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-d3b1b425-02ff-4358-983d-aae27ad8200d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d3b1b425-02ff-4358-983d-aae27ad8200d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ebb04bc6-e755-43f0-a490-42e3a34bb0e2">
  <button class="colab-df-quickchart" onclick="quickchart('df-ebb04bc6-e755-43f0-a490-42e3a34bb0e2')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ebb04bc6-e755-43f0-a490-42e3a34bb0e2 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
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


```python
numerical_features = ["Engine Size(L)",	"Cylinders", "Fuel Consumptions"]
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()
```





  <div id="df-21fe7ad5-751a-427c-8c9b-29b63a6bb3e3" class="colab-df-container">
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-21fe7ad5-751a-427c-8c9b-29b63a6bb3e3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-21fe7ad5-751a-427c-8c9b-29b63a6bb3e3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-21fe7ad5-751a-427c-8c9b-29b63a6bb3e3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f9f0455b-8024-43e3-b595-62a541b53bdc">
  <button class="colab-df-quickchart" onclick="quickchart('df-f9f0455b-8024-43e3-b595-62a541b53bdc')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f9f0455b-8024-43e3-b595-62a541b53bdc button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
X_train[numerical_features].describe().round(4)
```





  <div id="df-cb867550-a8d3-4aeb-a0ad-6c88d24dde53" class="colab-df-container">
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
      <td>-0.0000</td>
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
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-cb867550-a8d3-4aeb-a0ad-6c88d24dde53')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-cb867550-a8d3-4aeb-a0ad-6c88d24dde53 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-cb867550-a8d3-4aeb-a0ad-6c88d24dde53');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b5675aac-2209-4a24-90ef-3a7d94af1d74">
  <button class="colab-df-quickchart" onclick="quickchart('df-b5675aac-2209-4a24-90ef-3a7d94af1d74')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b5675aac-2209-4a24-90ef-3a7d94af1d74 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




# Model Development

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

Evaluasi model adalah proses untuk mengukur performa model prediktif dengan menggunakan data yang tidak dilibatkan dalam proses pelatihan. Salah satu metrik evaluasi yang umum digunakan untuk masalah regresi adalah **Mean Squared Error (MSE)**.

---

### **Mean Squared Error (MSE)**
MSE dihitung sebagai rata-rata kuadrat selisih antara nilai sebenarnya $$ y_i $$ dan nilai prediksi $$ \hat{y}_i $$. Rumusnya adalah:
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$
Keterangan:
- $$ n $$: Jumlah data
- $$ y_i $$: Nilai aktual pada data ke-\( i \)
- $$ \hat{y}_i $$: Nilai prediksi pada data ke-\( i \)

Nilai MSE yang lebih kecil menunjukkan model memiliki kemampuan prediksi yang lebih baik, karena kesalahan prediksi lebih kecil.

---  

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





  <div id="df-ed989526-c45b-496a-9458-556235f40c9c" class="colab-df-container">
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
      <td>0.094991</td>
      <td>0.12252</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>0.002306</td>
      <td>0.007852</td>
    </tr>
    <tr>
      <th>Boosting</th>
      <td>0.135352</td>
      <td>0.128702</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ed989526-c45b-496a-9458-556235f40c9c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ed989526-c45b-496a-9458-556235f40c9c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ed989526-c45b-496a-9458-556235f40c9c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-817fbdae-9fc6-44c9-9006-f3a5bbf6cd09">
  <button class="colab-df-quickchart" onclick="quickchart('df-817fbdae-9fc6-44c9-9006-f3a5bbf6cd09')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-817fbdae-9fc6-44c9-9006-f3a5bbf6cd09 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_4eee6097-0801-44e2-8a7e-3191d4450d9f">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('mse')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_4eee6097-0801-44e2-8a7e-3191d4450d9f button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('mse');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
```


    
![png](Predicitve_Analytic_Vehicle_Emissions_files/Predicitve_Analytic_Vehicle_Emissions_99_0.png)
    



```python
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true': y_test[:1]}

for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(0)

pd.DataFrame(pred_dict)
```





  <div id="df-f9e06c3e-243f-43ba-bdec-0805738fd365" class="colab-df-container">
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
      <td>285.0</td>
      <td>281.0</td>
      <td>280.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f9e06c3e-243f-43ba-bdec-0805738fd365')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f9e06c3e-243f-43ba-bdec-0805738fd365 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f9e06c3e-243f-43ba-bdec-0805738fd365');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




## Diskusi dan Interpretasi
1. MSE vs Hasil Prediksi:
Meskipun Random Forest memiliki MSE yang paling rendah, prediksinya pada contoh spesifik ini tidak lebih akurat daripada Boosting. Hal ini menunjukkan bahwa MSE mencerminkan performa rata-rata model, bukan performa pada sampel individual.

2. Boosting:
Meskipun memiliki MSE yang lebih tinggi secara keseluruhan, Boosting memberikan prediksi paling akurat untuk contoh ini. Ini bisa terjadi karena model Boosting mungkin lebih baik dalam menangkap pola tertentu tetapi kurang optimal pada data secara keseluruhan.

3. KNN:
Prediksi KNN pada sampel ini lebih jauh dari nilai sebenarnya, mendukung hasil MSE yang menunjukkan performanya lebih buruk dibandingkan RF.

## Kesimpulan
Untuk performa keseluruhan, Random Forest tetap menjadi model terbaik berdasarkan MSE yang rendah.

# Persiapkan Laporan


```python
# !pip install nbconvert
!jupyter nbconvert --to markdown /content/gambar.ipynb
```

    [NbConvertApp] WARNING | pattern '/content/gambar.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    



```python
!zip -r gambar_files.zip gambar_files
```

    	zip warning: name not matched: gambar_files
    
    zip error: Nothing to do! (try: zip -r gambar_files.zip . -i gambar_files)


# Referensi

> 1. Anggraeni, D. N. (2017). Kajian Emisi CO2 Dari Kendaraan Bermotor Di Kampus I Universitas Brawijaya Dan Kampus I Universitas Negeri Malang. Retrieved from https://api.semanticscholar.org/CorpusID:127584962

> 2. Sudarti, S., Yushardi, Y., & Kasanah, N. (2022). Analisis Potensi Emisi CO2 Oleh Berbagai Jenis Kendaraan Bermotor di Jalan Raya Kemantren Kabupaten Sidoarjo. *Jurnal Sumberdaya Alam dan Lingkungan*. Retrieved from https://api.semanticscholar.org/CorpusID:252253520

> 3. Kusumawardani, D., & Navastara, A. M. (2018). Analisis Besaran Emisi Gas CO2 Kendaraan Bermotor Pada Kawasan Industri SIER Surabaya. *Jurnal Teknik ITS*, 6, 399-402. Retrieved from https://api.semanticscholar.org/CorpusID:115275286

> 4. Kusumawardani, D. (2017). Arahan Penyediaan Ruang Terbuka Hijau Dalam Menyerap Emisi Gas Co2 Kendaraan Bermotor Pada Kawasan Industri Sier, Surabaya. Retrieved from https://api.semanticscholar.org/CorpusID:169900032

> 5. Amelia, C. R., Samadikun, B. P., & Huboyo, H. S. (2017). Analisis Shifting Penggunaan Moda Kendaraan Bermotor Ke Kereta Api Terhadap Penurunan Emisi Gas Rumah Kaca (CO2, CH4, dan N2O) Studi Kasus: Daerah Operasional VIII Surabaya. Retrieved from https://api.semanticscholar.org/CorpusID:114581634

> 6. Paradizsa, I. (2023). Analisis Kebijakan Pengendalian Polusi melalui Uji Emisi Kendaraan Bermotor Berbahan Bakar Minyak (BBM) di Wilayah DKI Jakarta. *Jurnal Enviscience*. Retrieved from https://api.semanticscholar.org/CorpusID:268714970

> 7. Buanawati, T. T., Huboyo, H. S., & Samadikun, B. P. (2017). Estimasi Emisi Pencemar Udara Konvensional (SOx, NOx, CO, dan PM) Kendaraan Pribadi Berdasarkan Metode International Vehicle Emission (IVE) di Beberapa Ruas Jalan Kota Semarang. Retrieved from https://api.semanticscholar.org/CorpusID:117120421

