
# Identifying and Categorizing Offensive Language in Social Media

Zadanie ze strony:

https://sites.google.com/site/offensevalsharedtask/offenseval2019

Rozwiązania zadania a i zadania b są w dwóch plikach:

Task a: [task_a_w2vec.ipynb] (task_a_w2vec.ipynb)

Task b: [task_b_w2vec.ipynb] (task_b_w2vec.ipynb)


## Środowisko

Karta i sterowniki nvidia +  cuda.

Środowisko conda, wszystkie biblioteki w pliku torch_cuda.yml.

W celu przyspieszenia obliczeń używam CUDA.

Takie samo środowisko można użyć komend:

    conda create -n torch
    conda activate torch
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
    conda install -c conda-forge pandas matplotlib numpy scikit-learn spacy ipython
    conda install -c conda-forge ipython notebook jupyterlab jupytext


## Proces wczytywania danych, tworzenia modelu i nauki od początku do końca

Poniższy proces dotyczy obu zadań.

Najpierw importuję biblioteki i sprawdzam czy działa cuda-pytorch.

Ładuję dane treningowe i testowe do osobnych obiektów typu DataFrame w Pandas.

W przypadku danych testowych łącze dane wejściowe (tweety) z etykietami (różne pliki) do jednego obiektu DataFrame.

Potem procesuję każdego tweeta w Spacy, który rozbija na obiekty typu Spacy, które zapisuję w nowej kolumnie treningowego i testowego DataFrame.

Do kolejnej kolumny(lemmas) zapisuję lematy, uzyskane za pomocą atrybutu .lemma_ z poprzednio utworzonej kolumny. Pomijam lematy z @, #, stop words itp.

Tworzę słownik dict_lemmas (słowo: unikalna liczba) dla wszystkich słów z obu źródeł danych.

Każdemu Tweetowi przypisuje number odpowiadający lematowi ze słownika dict_lemmas. Te numery zapisuję jako lista o ustalonej szerokości najdłuższego tweeta, przesuniętej w prawo, z zerami symbolizującymi puste miejsca. To zapisuje w nowej komórce numbers.

Generuję (albo wczytuję) wektory embeddingów ze Spacy (szerokość, czy wymiar 96) w trochę okrężny sposób, iterując przez słownik dict_lemmas i pytając Spacy o lemat dla danego słowa i zapisując do listy embeddings. Potem tą listę konwertuję na tensor Pytorchowy. Chodzi o to, żeby wiersz tensora embedingów odpowiadał liczbie ze słownika dict_lemmas.

Potem definiuję model w Pytorchu. Modelowi przekazywane są:

* ilość (szerokość) danych wejściowych - odpowiada najdłuższemu tweetowi z danych treningowych lub testowych.
* ilość neuronów pierwszej i drugiej warstwy ukrytej (ewentualnie trzeciej jeśli ma przypisaną liczbę).
* ilość danych wyjściowych - klasyfikacja binarna więc jeden neuron wyjściowy.


### Ustawienia sieci neuronowej, tworzenie instancji modelu, funkcja straty, optymalizator, uczenie i wyświetlanie rezultatów

Ustawiam funkcję straty na BCEloss, czyli Binary Cross Entropy, ze względu na to że mamy do czynienia z klasyfikacją(Cross Entropy) binarną(Binary). Jako optymalizator ustawiam Adam.

Potem iteruję przez liczbę epok, a w tym przez batche (partie danych) i przechodzę przez cały proces uczenia. W międzyczasie dla każdej epoki zapamiętuję wartość straty i accuracy. Nie wiem czy typowo czy nietypowo po każdej epoce nauki sprawdzam wartość straty i accuracy dla danych testowych (oczywiście bez obliczania gradientu).


## Ustawienia sieci neuronowej

Zadania a i b są podobne i sieci dla obu tych zadań są podobne.

Po warstwie embedingów mamy 2 lub 3 ukryte warstwy liniowe (h1, h2, …), czyli każdy neuron z poprzedniej warstwy jest połączony na każdym z kolejnej (pomijając warstwę dropout).

Przy 2 ukrytych warstwach, przy nawet małej liczbie neuronów w tych warstwach, sieć uczy się całkiem nieźle. Im więcej neuronów w warstwie h1 i h2 tym model do pewnego momentu uczy się szybciej (kiedy reszta parametrów pozostanie taka sama). Learning rate ustawiony na 0.001 wydaje się dobrym środkiem. Eksperymentowałem z batch size od 50 do 1000, i nie ma wielkich różnic w większości konfiguracji. Czasem wydawało się, że mniejszy batch size daje lepsze rezultaty, ale przy innych ustawieniach nie robiło to już różnicy.

W zadaniu a występuje następujący problem. Dane testowe mają najlepsze rezultaty na samym początku, przed tym, aż sieć się wytrenuje na danych treningowych. Oczywiście jest to normalne, że im bardziej sieć douczy się do danych treningowych, w pewnym momencie sieć coraz gorzej przewiduje dane testowe. Ale tutaj wygląda to tak, że po 1-2 epokach dane testowe mają najlepszą accuracy, a dane treningowe dużo gorsze. a potem dla danych treningowych rośnie a dla testowych fluktuuje, w zależności od hiperparametrów.

Dla task_a h1=2, h2=3 co ciekawe daje całkiem niezłe rezultaty dla danych testowych, natomiast treningowe, jeśli chodzi o accuracy, nie wychodzą dużo powyżej 70% i pozostają niższe niż testowe. Po dodawaniu pojedynczych neuronów do h1 czy h2 sieć albo wariuje (nic się nie uczy, tzn rozpoznawanie spada) albo albo zbyt bardzo dostosowuje się do danych treningowych a kategoryzacja danych testowych albo fluktuuje albo bardzo spada.

Celem byłoby, żeby współczynniki dla danych treningowych się polepszały, a dla danych testowych przynajmniej podwyższyły się w stosunku do epoki 1-2. Chodzi mi o to, żeby jednocześnie sieć się nauczyła i dostosowała do danych treningowych a jednocześnie jakoś wykrywała te testowe.

Końcowo zdecydowałem się na h1=2, h2=3 i obu zadaniach. Jeśli chodzi o zadanie a wydaję mi się, że trzeba byłoby zatrzymać nauczanie pomiędzy epoką 75 a 90 i takiego modelu używać.

Zadanie b jest prostsze, tzn dane testowe i treningowe wydają się bardziej zbieżne i różne konfiguracje sieci dają niezłe efekty. Po pewnym czasie wydaje się, że sieć przestaje się po prostu uczyć i zbiega konkretnych parametrów wewnętrznych w sieci.

Wykresy i tabelki znajdują się w jupyter notebooku.
