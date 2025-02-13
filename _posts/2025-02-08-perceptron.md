---
title: Perceptron
date: 2025-02-08
categories: [Computational Neuroscience]
tags: [perceptron]
description: Tek katmanlı Perceptron modeline giriş
toc: false
---


# Biyolojik Nöronlardan Yapay Sinir Ağlarına : Perceptron

Merhaba!
[Önceki yazımın](https://zypchn.github.io/posts/intro-to-brain-science/) devamı olan bu yazıda, biyolojik nöronlardan ilham alınarak tasarlanan Perceptron modelinden bahsedeceğim. Bu yazıdan sonra hesaplamalı nörobilime giriş yapacağımız için öncesinde yapay sinir ağlarının temel birimi hakkında bilgi sahibi olunması gerektiğini düşünüyorum.
Keyifli okumalar!

<br/>

## Perceptron Nedir?
Perceptron, tek katmanlı bir yapay sinir ağının temel birimi olan denetimli bir öğrenme algoritmasıdır. İkili sınıflandırma için uygundur, yani giriş değerlerini iki kategoriye ayırır: 0/1 veya -1/1 (kaynaklarda farklılık göstermektedir).

Perceptron algoritması ikiye ayrılır:

 1. **Tek Katmanlı Perceptron** : Verinin tek bir çizgiyle bölünebildiği senaryolar için uygundur. Karmaşık problemler için uygun değildir. Aşağıda algoritma için uygun bir verinin örneği verilmiştir:
![enter image description here](https://automaticaddison.com/wp-content/uploads/2019/07/linearly-separable.png)

2. **Çok-Katmanlı Perceptron** : Birden çok Perceptron katmanı içerdiği için veri işlem yeteneği daha yüksektir. Verideki daha karmaşık ilişkileri çözebilir.

Ben bu yazımda Tek-Katmanlı Perceptron modeli üzerinden ilerleyeceğim.

<br/>

## Perceptron'un Bileşenleri
Bir yapay sinir ağı dört adet bölümden oluşmaktadır: giriş değerleri (input features), ağırlıklar ve sapma (weights and bias), ağırlıklı toplam (weighted sum) ve aktivasyon fonksiyonu (activation function). Bu bileşenleri daha yakından inceleyelim:

**Giriş Değerleri :** Her giriş (input) verisinin bir takım özellikleri vardır ve bu özellikleri temsil etmek için sayısal değerler atanmıştır. 
[ $x_1 = 3.8, x_2 = 5.1,  ..., x_n = 2.9$ ]
Buradaki her $x_n$ değeri bir giriş değeri olarak adlandırılır. Giriş değerlerinin sayısı verideki özelliklerin sayısına bağlıdır.

**Ağırlıklar ve Sapma :** Giriş verisindeki her bir özelliğin çıkış verisi üzerindeki etkisini belirleyen sayısal değerlere *ağırlık* denmektedir.  
[ $w_1=2, w_2=8.5, ...,  w_n = -1.4$ ] 
Ağırlıklar, algoritmanın eğitim süresi boyunca değiştirilerek optimal değerleri bulunmaya çalışılır.
*Sapma* değeri ise algoritmanın giriş değerlerinden bağımsız değişiklikler yapmasına olanak tanıyan skalar değerdir. Genelde $b$ harfiyle (bias) ifade edilir.

**Ağırlıklı Toplam :** Ağırlıklı toplam değeri, her giriş değerinin kendisine ait ağırlık değeriyle çarpılması ve sonuçların toplanmasıyla elde edilen değerdir. 
$z = w_1x_1 + w_2x_2 + w_3x_3 + ..., w_nx_n$

**Aktivasyon Fonksiyonu :** Ağırlıklı toplamda elde edilen değeri girdi olarak alır ve çıkış değerini verir. 
Aktivasyon fonksiyonu problemden probleme veya algoritmadan algoritmaya değişiklik gösterebilir. Perceptronlar, *Heaviside Basamak Fonksiyonu'nu (binary step)* kullanmaktadır ve şöyle tanımlanmıştır: 
$$0 \quad if \quad z < Threshold \brace 
1 \quad if \quad z \geq Threshold$$

Buradaki *z* ağırlıklı toplamı, *Threshold* ise eşik değerini ifade etmektedir. Eşik değer probleme göre değişiklik gösterebilir.

Perceptron modelinin bileşenlerini öğrendik, bir de görsel olarak nasıl modellendiğine bakalım:
![perceptron](https://media.geeksforgeeks.org/wp-content/uploads/20221219111343/Single-Layer-Perceptron.png)

Tasarım size de tanıdık gelmedi mi? Sanki şuna benziyor:
![neuron](https://doktorfizik.com/wp-content/uploads/2019/12/n%C3%B6ron.png)

İki görseli birleştirecek olursak: <br/>
![neuron + perceptron](https://miro.medium.com/v2/resize:fit:828/format:webp/1*FLoEcD4bWRw6Zno32uFwuw.png)

Gördüğünüz gibi Perceptron ve biyolojik nöronlar tasarım olarak oldukça benziyor. 

<br/>

## Perceptron Nasıl Çalışır : Örnek Senaryo
Yukarıda anlatılanları pekiştirmek için gerçek dünyadan bir senaryoyla örnek verelim. Diyelim ki bir giyim mağazasındasınız. Üzerinde en sevdiğiniz Pokemon'un çizimi olan bir tişört var ve onu alıp almayacağınıza karar vermek istiyorsunuz. Beyniniz *evet* ya da *hayır* olarak karar verecektir, yani *1* veya *0*. Karar verebilmeniz için her özelliğin bir değeri ve ağırlığı olmalıdır, bunun için aşağıdaki tablolara bakalım:

| Giriş Özelliği | Giriş Değeri | Ağırlık |
|--|--|--|
| Fiyat | $x_1 = 400$ | $w_1 = -0.02$ |
| Dizayn | $x_2 = 5$ | $w_2 = 2$ |
| Kumaş Kalitesi | $x_3 = 3.5$ | $w_3 = 1.4$ |

<br/>

| Sapma | Eşik Değer  |
|--|--|
| $b=0.5$ | $T=8$ |


Burada *Dizayn* ve *Kumaş Kalitesi* özelliklerine 5 üzerinden puan verdiğinizi varsayalım. *Fiyat* ise ürünün gerçek fiyatı. Dikkat ettiyseniz *Fiyat* özelliğinin ağırlığı negatif bir değer. Demek ki fiyatın yüksek olması kararımızı olumsuz etkiliyor. Kalan özelliklerde ise durum tam tersi, yüksek puan kararımızı olumlu etkiliyor. Ağırlık değerleri probleme veya hangi özelliğe ne kadar önem vermek istediğinize göre değişebilir.

Giriş, ağırlık, sapma ve eşik değer tanımlandığına göre ağırlıklı toplam hesaplamaya geçebiliriz:
$z = w_1x_1 + w_2x_2 + w_3x_3 + b$
$= (-0.02 \times 400) + (2 \times 5.0) + (1.4 \times 3.5) + 0.5$
$= (-8) + (10) + (4.9) + (0.5) = 7.4$

Daha sonra elde ettiğimiz değeri $H(z)$ basamak fonksiyonuna verelim:
$$0 \quad if \quad z < 8 \brace 
1 \quad if \quad z \geq 8$$
Buradan, $H(7.4) = 0$ değeri geliyor.  Sonuç olarak *hayır* cevabını alıyoruz, bye Charizard 😭

Burada optimal parametreleri önceden bildiğimiz için modelimiz direkt karar verebildi. Peki parametreleri nasıl bulduk? Bunun için Perceptron'un eğitim sürecine bakmamız lazım.

<br/>

## Perceptron Eğitim Süreci : Kod Örneği

:warning: Bu bölümdeki kodları anlamak için makine öğrenme ve PyTorch bilgisi gerekmektedir.

<br/>
İlk önce gerekli kütüphaneleri import edelim:

    import torch
    import torch.nn as nn
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
 
 <br/>

Burada hazır veri seti kullanmak yerine kendi veri setimizi oluşturacağız. Bunun için *sklearn* kütüphanesinden *make_blobs* fonksiyonunu kullanıyoruz:

    X, y = make_blobs(
    n_samples=1000,		# giriş verilerinin sayısı
    n_features=2,		# her giriş verisinin özellik sayısı
    centers=2,
    cluster_std=3,
    random_state=23
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2,			# tüm verilerin %20'sini test için ayırıyoruz
    random_state=23, 		
    shuffle=True			# verileri bölmeden önce karıştırıyoruz
    )

<br/>

Veri setimizi oluşturduktan sonra algoritmaya uygun hale getirmek için veri ön işleme (data preprocessing) yapmamız gerekiyor:

	# giriş verilerini sıfır ortalama ve birim varyansa sahip olacak şekilde ölçeklendiriyoruz
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

	# tüm verileri PyTorch tensörlerine dönüştürüyoruz
	X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=False)
	X_test = torch.tensor(X_test, dtype=torch.float32, requires_grad=False)
	y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)
	y_train = torch.tensor(y_train, dtype=torch.float32, requires_grad=False)
	
	# y verilerini modelin çıktı olarak vereceği tensörle aynı formata getiriyoruz
	y_train = y_train.reshape(-1, 1)
	y_test = y_test.reshape(-1, 1)

	torch.manual_seed(42)
	
<br/>

Veri ön işleme adımımızı tamamladık. Şimdi Perceptron sınıfımızı tanımlayacağız:

    class Perceptron(nn.Module):
	    def __init__(self, num_inputs):
		    super(Perceptron, self).__init__()
		    self.linear = nn.Linear(num_inputs, 1)
		
		# Heaviside Basamak Fonksiyonu
		def binary_step_func(self, Z):
			Class = []
			for z in Z:
				if x >= thr:
					Class.append(1)
				else:
					Class.append(0)
			return torch.tensor(Class)
			
		def forward(self, x):
			Z = self.linear(x)
			return self.binary_step_func(Z)

<br/>

Perceptron sınıfımızı oluşturduk. Şimdi objemizi ve maliyet (cost) fonksiyonumuzu tanımlayalım:

    perceptron = Perceptron(num_inputs=X_train.shape[1])

	def loss_func(y_pred, Y):
		cost = y_pred - Y
		return cost

<br/>

Şimdi eğitim aşamasına geçebiliriz. Bunun için *öğrenim değeri (learning rate / step size)* ile *epoch sayısı* değerlerini tanımlayalım ve ardından eğitim döngümüzü yazalım:

    learning_date = 1e-3 	# 0.001
    num_epochs = 10

	for epoch in range(num_epochs):
		Losses = 0
		for Input, Class in zip(X_train, y_train):
			prediction = perceptron(Input)
			error = loss(prediction, Class)
			Losses += error

			# Perceptron Öğrenme Kuralı
		
			# Model Parametreleri
			w = perceptron.linear.weight	# ağırlıklar
			b = perceptron.linear.bias		# sapma
		
			# Güncellenen Model Parametreleri
			w = w - learning_rate * error * Input
			b = b - learning_rate * error

			# Güncellenen parametreleri modele geri gönderiyoruz
			perceptron.linear.weight = nn.Parameter(w)
			perceptron.linear.bias = nn.Parameter(b)
	  print('Epoch [{}/{}], weight:{}, bias:{} Loss: {:.4f}'.format(
	      epoch+1,num_epochs,
	      w.detach().numpy(),
	      b.detach().numpy(),
	      Losses.item()))

> Epoch [1/10], weight:[[0.6278487 0.24864984]], bias:[-0.08365549] Loss: -82.0000 <br/>
> Epoch [2/10], weight:[[0.6140272 0.10244947]], bias:[-0.05165547] Loss: -32.0000 <br/>
> Epoch [3/10], weight:[[0.5870175 0.0165318]], bias:[-0.0316555] Loss: -20.0000 <br/>
> Epoch [4/10], weight:[[0.558535 -0.04593931]], bias:[-0.0236555] Loss: -8.0000 <br/>
> Epoch [5/10], weight:[[0.5315978 -0.08121765]], bias:[-0.0216555] Loss: -2.0000 <br/>
> Epoch [6/10], weight:[[0.5053688 -0.11094457]], bias:[-0.0216555] Loss: 0.0000 <br/>
> Epoch [7/10], weight:[[0.48110715 -0.1318747 ]], bias:[-0.0186555] Loss: -3.0000 <br/>
> Epoch [8/10], weight:[[0.45835102 -0.14691873]], bias:[-0.0166555] Loss: -2.0000 <br/>
> Epoch [9/10], weight:[[0.43633538 -0.15988223]], bias:[-0.0126555] Loss: -4.0000 <br/>
> Epoch [10/10], weight:[[0.41519952 -0.17041822]], bias:[-0.0106555] Loss: -2.0000 <br/>


Burada tüm epoch'lar için model ağırlıkları (weight), sapma (bias) ve kayıp (loss) değerlerini görebilirsiniz. (Veri seti ve parametreler rastgele oluşturulduğu için alacağınız çıktı farklılık gösterebilir.)

Modelimiz eğitildiğine göre test aşamasına geçebiliriz. Hatırlarsanız en başta veri setimizin %20'sini ayırmıştık. Şimdi onu kullanacağız:

    pred = perceptron(X_test)	# tahmin edilen değerler
	
	accuracy = (pred == y_test[:, 0]).float().mean()	# doğruluk metriği
	print("Test Verisinin Doğruluk Değeri:", accuracy.item())

> Test Verisinin Doğruluk Değeri: 0.9599999785423279

Doğruluk değeri yine parametre ve verilerin rassallığından dolayı sizde farklılık gösterebilir.

<br/>

Okuduğunuz için teşekkür ederim 🤓 <br/>
Bir sonraki yazımda biyolojik nöronları daha gerçekçi bir şekilde uygulamaya koyan modellerden bahsedeceğim. <br/>
Görüşmek üzere 👋

<br/>
<br/>

### Kaynaklar :
- https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/
