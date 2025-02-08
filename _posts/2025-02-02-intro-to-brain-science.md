---
title: Beyin Bilimine Giriş
date: 2025-02-01
categories: [Computational Neuroscience]
tags: [brain, neuron, synapses]
description: Hesaplamalı nörobilim öğrenmek için beyin ve işleyişi hakkında genel bir tanıtım
toc: false
---

Merhaba! <br/>
Hesaplamalı nörobilim hakkında yazılar paylaşacağım serimin ilk yazısında beyin ve bilinç hakkında genel bilgiler verdim. *Hesaplama* kısmına geçmeden önce *nörobilim* hakkında fikir sahibi olunmasının ilerleyen süreçte faydalı olacağını düşünüyorum. <br/>
Keyifli okumalar!

## Beynin Bölgeleri
İnsan beyni ortalama 1400-1500 gram ağırlıktadır, bu da toplam vücut ağırlığının %2'sine denk gelmektedir. Buna rağmen vücudumuza giren oksijenin %20 kadarını harcıyor. Harcadığı enerjiyle bakılırsa önemli şeyler yapıyor olmalı, değil mi? 
Beyin her ne kadar tek bir organ olsa da onu tek bir işleve indirgemek yanlıştır. Onun yerine beyni bölgesel olarak incelemek daha doğrudur. Aşağıdaki şemaya bakarak beynin ana bölgelerini tanıyalım:
![beyin-sema](https://www.dunyadanismanlikmerkezi.com/wp-content/uploads/2022/07/beyin-bolgeleri.jpg)

 - **Frontal Lob** : Bilinçi düşünmeden sorumlu olan bölge.
 - **Parietal Lob** : Duyuları işleme, şekil ve renkleri algılama, uzaysal algı, görme algısı ve aritmetik yetenekleri kontrol eder.
 - **Oksipital Lob** : Görme duyusunun işlenmesinden sorumlu olan bölge.
 - **Temporal Lob** : Görsel hafıza, dil anlama, işitme duyusu, ağrı ve duygusal ilişkilendirmeden sorumlu olan bölge.
 - **Beyincik** : Dengeyi sağlayan bazı motor fonksiyonları (yürüme, ayakta durma, vb.) kontrol eden kısımdır.
 - **Omurilik** : Beyinden uyarıları taşıyan sinir dokusudur. Birçok refleksi başlatan ve koordine eden merkezdir.

Görüldüğü üzere tek bir organda birden fazla ve oldukça çeşitli işlemler gerçekleşmektedir. Peki bu işlemler nasıl gerçekleşiyor? Bunun için beynin temel çalışma ünitelerine bakmamız lazım: nöronlar.

## Beynin Yapı Taşları : Nöronlar
Nöronlar, sinir sistemini oluşturan sinir hücreleridir. Görevleri sinirsel uyarıları ileterek vücuttan beyne, beyinden vücuda veya beynin kendi içinde sinyal (bilgi) taşımaktır. 
![noron-sema](https://doktorfizik.com/wp-content/uploads/2019/12/n%C3%B6ron.png)
Nöronun yapısı basit olarak yukarıda verilmiştir. Görüldüğü üzere en basit modellemede bile birçok bölüm bulunuyor fakat biz 2 kısıma odaklanacağız: Dendrit ve Akson

 - **Dendrit** : Nöronda bulunan input (giriş) port olarak düşünülebilir. Yani nöronun giriş sinyalini aldığı yerdir.
 - **Akson** : Nöronda bulunan output (çıkış) port olarak düşünülebilir. Yani nöronun çıkış sinyalini verdiği yerdir.

Hücre zarının içi, dışına göre negatiftir bu da hücre zarının elektriksel olarak yüklü olmasına sebep olur. Hücre zarının iki yanı arasındaki potansiyel farklılık, *zar potansiyeli* olarak (bazı kaynaklarda membran potansiyeli/iletkenliği olarak da geçer) adlandırılır ve *mV* olarak ölçülür.

Nöronun zar potansiyeli değiştiğinde nöron bu değişikliğe *aksiyon potansiyeli* adı verilen bir voltaj sapmasıyla cevap 
verir. Zar potansiyelindeki sapma negatif veya pozitif olabilir. Nöronlar uğradıkları sapmaya göre sınıflandırılır: pozitif sapmaya uğrayanlar uyarıcı, negatif sapmaya uğrayanlar engelleyici (inhibitör) adını alır.


## Nöronlar Arası Köprü : Sinapslar
Sinapslar, nöronlar arası iletişimi sağlayan özelleşmiş boşluklardır. İki adet sinaps türü vardır: elektriksel ve kimyasal.

- **Elektriksel Sinaps** : Hücre zarlarının birbirine çok yaklaştığı bölgelerde hücreler arası kanallar aracılığıyla gerçekleşir. 
- **Kimyasal Sinaps** : Akson ve Dendrit arasındaki bağlantı özel bir kimyasal (nörotransmiter) madde ile sağlanır. Burada kimyasalı veren nöron pre-sinaptik, alan nöron ise post-sinaptik nöron olarak adlandırılır. Post sinaptik nörondaki reseptörlere bağlanan nörotransmiter madde, bağlandığı nöronun zar potansiyelinde bir sapmaya neden olur.

<br/>
Nöron ve sinapsların neler olduğunu öğredik. Bir önceki sorumuza dönelim: beyin kendi içinde bölümlere nasıl ayrılıyor? 

![connectivity-matrix](https://earimediaprodweb.azurewebsites.net/Api/v1/Multimedia/65b336d6-dc73-4e09-a1d8-caba60abf2e5/Rendition/low-res/Content/Public)
Yukarıdaki görsel bir fare beyninin bağlantı matrisini (connectivity matrix) göstermektedir. Bağlantı matrisleri beynin yapı ve işleyişini temsil eden matrislerdir. Kırmızı yüksek, sarı orta, mavi ise düşük derecede bağlantıları göstermektedir. Buradan görüyoruz ki beynin bölümlerindeki sinaptik bağlantılar rastgele değildir. Her bölüm en fazla bağlantıyı yine kendi içerisinde kurmaktadır. Bu bilgi işleme için çok kritiktir, çünkü gereksiz ve düzensiz bağlantılar işlem sürecini yavaşlatabilir veya bilişel yük yaratabilir. Bu da beynin iletilen sinyalleri rastgele işlemediğini, yani bölümlere özgü işlediği anlamına gelmektedir.

Bölgelerarası bağlantısallık türü, büyük ölçüde genetik altyapıyla ilgilidir, bir başka deyişle türe özgüdür. Örneğin insanda yaklaşık 86 milyar nöron varken, fillerde bu sayı 257 milyara çıkmaktadır. Neredeyse 3 kat fazla nörona sahip fillerin zekası insanlarınkine yakın bile değildir. Çünkü insan beyninde her nöronda ortalama 10 bin sinaps bulunurken fillerde bu sayı çok daha azdır. Bu sayı bir nöronun ortalama 10 bin girdi alması ve 10 bin çıktı göndermesi anlamına gelir. İnanılmaz bir işlem gücü, değil mi? Buradan yoğun bağlantının *çoğu zaman* yüksek zekayla sonuçlanacağı çıkarımını yapabiliriz. 

Yaptığımız çıkarım akıllara şu soruyu getirebilir: Nöron sayısı arttıkça sinaps potansiyeli ve buna bağlı olarak sinaps yoğunluğu artmaz mı? 
Bu mantıklı bir çıkarımdır, çünkü daha fazla nöron, daha fazla bağlantı kurma olasılığı yaratır. Fakat zekayı etkileyen bir diğer faktör bağlantıların kalitesi ve adaptasyon yeteneğidir. Burada yeni bir kavramla tanışıyoruz: **sinaptik plastisite**. Plastisite, öğrenmeye ve deneyime bağlı olarak değişebilme yeteneği olarak tanımlanabilir. Bahsedilen değişim sinapsların güçlenip zayıflamasıdır ve zekada kritik bir rol oynar. 

Yüksek zeka için, sinaps yoğunluğu önemli bir faktör olsa da tek başına yeterli değildir. Önemli olan bu bağlantıların nasıl düzenlendiği ve değişime açık olup olmadığıdır. Beyin eğer işlevsel ve optimize edilmiş bağlantılar kurabilirse zeka artabilir. 

Okuduğunuz için teşekkür ederim 🤓 <br/>
Bir sonraki yazımda biyolojik nöronlardan esinlenerek modellenen yapay sinir ağlarından bahsedeceğim. <br/>
Görüşmek üzere 👋

<br/>
<br/>

### Kaynaklar :
- [https://youtube.com/watch?v=mZkujcMW1uI](https://youtube.com/watch?v=mZkujcMW1uI)
- https://evrimagaci.org/sinapslar-ve-iletisim-hucreden-hucreye-iletisimde-elektriksel-sinapslarin-rolu-nedir-9093?srsltid=AfmBOoqQeNYpngYKoL2FhDD6x9boo2j9QJqCSW7zdJoakf_-XZ_174be