# CipherTypeFinder
June 2026

Program created for Caramba Team, Loria.

The aim of this program is to get from a raw photo of an encrypted manuscript, it's cipher type included into these 6 categories: 

# Cipher Type Finder — Architecture Schema

## 1. Input

- **Ciphered document (.jpg)**

## 2. Preprocessing

### 2.1 Image Preprocessing

1. **Binarization**

2. **Skew Correction**

3. **Morpho Cleaning**

4. **Linear Segmentation**

5. **Linear Noise Filtering**

   

### 2.2 Data Preprocessing

- =>**Document lines** 

  - Space/dot based tokenization + Character/token count

- **Connected Component Analysis** → produces **Document characters**

  - Character size analysis
  - Character occurence per glyphe count (including spacing/dots)
  - Character clustering

  

------

## 3. Feature Vectors

Outputs of preprocessing feed into two vector types:

- **Vector Numerical Data**

  Made from:

  - Characters frequencies
  - Characters occurences value
  - Glyphe number into the alphabet
  - Character size variance
  - Characters per pool

- **Vector Image**

  Made from:

  - Manuscript's lines preprocessed

------

## 4. Neural Network Architecture

### Fusion

- **TIP or MMCL — MERGE** (combine image vector and tabular vector)

=> Flatten Layer

### Processing

**FeedForward Neural Network**

- Activation function: **ReLU or Tanh**

=> Hidden Layer

### Finalization

- Activation function: **SoftMax**

- Loss Function: **Categorical Cross Entropy**

- Backpropagation function / Optimization: **Adam**

  => Output layer

------

## 5. Output: Cipher Type (Classification)

- Not encrypted
- Machine
- Homophonic
- Transposition
- Code book
- Polyalphabetic
- Simple / Monoalphabetic

Each class outputs a **Probability P**.

------

## Legend Recap (Classes)

```
Output: Cipher Type
├── Not encrypted
├── Machine
├── Homophonic
├── Transposition
├── Code book
├── Polyalphabetic
└── Simple / Monoalphabetic
```

<img src="/home/mae/Documents/idmc/master1/internship/caramba/shema_architecture_cipherTypeFinder.png" alt="shema_architecture_cipherTypeFinder" style="zoom:200%;" />shema_architecture_cipherTypeFinder



