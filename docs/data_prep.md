# Data Preparation Stage

- Convert the data into train.tsv and test.tsv in 70:30 ratio

```
data.xml

    |-train.tsv
    |- test.tsv
```

- We are choosing only __three__ tags in the xml data.
1. Row ID
2. Title and body
3. Tags (stackoverflow tags specific to python)

|Tags| feature names|
|-|-|
|row ID|row ID|
|title and body|text|
|stackoverflow tags| Label- Python|