# Современные подходы к постквантовой криптографии

В репозитории представлена реализация двух схем:

- многомерная криптография с использования пермутационных полиномов (`many_dim.py`)
- некоммутативный протокол с использованием матриц над конечным полем (`matrix.py`)


## Использование матричного шифрования

1. Генерация ключей
```
python3 matrix.py --dim {{N}} keygen [[--keydir {{KEYDIR}} ]]
```

2. Шифрование
```
python3 matrix.py --dim {{N}} encrypt --msg {{MSG}} [[--out {{OUTFILE}} ]] [[--keydir {{KEYDIR}} ]]
```

3. Дешифрование
```
python3 matrix.py --dim {{N}} decrypt --infile {{INFILE}} [[--out {{OUTFILE}} ]] [[--keydir {{KEYDIR}} ]]
```