# DeepLearningReport
# 🫁 Akciğer X-Ray Görüntülerinden Hastalık Sınıflandırma
## 📌 Proje Amacı
Bu proje, COVID-19 Radyografi Veritabanı kullanılarak göğüs röntgeni (X-ray) görüntülerinin sınıflandırılması için bir derin öğrenme modeli geliştirmeyi ve değerlendirmeyi amaçlar. Proje kapsamında, dört farklı sınıf (COVID-19, Normal, Viral Pnömoni ve Akciğer Opaklığı) arasındaki ayrımı yapmak için bir Evrişimsel Sinir Ağı (CNN) mimarisi eğitilmiştir.

## 📊 Veri Seti Hakkında Bilgi
Veri seti linki:https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Veri seti, toplam 21.165 göğüs röntgeni görüntüsünden oluşmakta olup, farklı hastaneler ve kaynaklardan derlenmiş, çeşitli kalite ve görüntüleme koşullarını içermektedir. Bu çeşitlilik, modelin farklı hasta profilleri ve görüntüleme senaryolarına karşı daha genelleştirilebilir ve güvenilir tahminler yapabilmesini sağlamaktadır. Dolayısıyla veri seti, COVID-19 ve diğer akciğer hastalıklarının tespitine yönelik araştırmalar için uygun bir temel sunmaktadır.Dört ana kategoride etiketlenmiştir:

🦠 COVID-19: COVID-19 vakalarına ait görüntüler.

✅ Normal: Sağlıklı bireylere ait görüntüler.

🫁 Viral Pnömoni: Viral pnömoni vakalarına ait görüntüler.

🌫️ Akciğer Opaklığı (Lung Opacity): Akciğerlerde opaklık gösteren görüntüler, genellikle COVID-19 gibi hastalıklarla ilişkilidir.

Veri setindeki sınıf dağılımı görselleştirilmiştir ve modelin dengesiz sınıflarla başa çıkabilmesi için çeşitli veri artırma (data augmentation) teknikleri uygulanmıştır.

# 🛠️ Kullanılan Yöntemler
Bu proje, göğüs röntgeni görüntülerinin sınıflandırılması için modern derin öğrenme yaklaşımlarını bir araya getirmektedir. Kullanılan temel yöntemler ve teknikler aşağıda detaylandırılmıştır:

## 1. Veri Artırma (Data Augmentation) ve Ön İşleme
Modelin genelleme yeteneğini artırmak ve küçük veri setinin etkilerini azaltmak için eğitim aşamasında yoğun veri artırma teknikleri uygulanmıştır.

🔹*Standartlaştırma (Normalization):* Görüntüler, ImageNet ortalama ve standart sapma değerleri kullanılarak normalize edilmiştir.
🔹*Boyutlandırma:* Tüm görüntüler 224×224 piksel boyutuna getirilmiştir.
🔹*Artırma Teknikleri:*
   -Rastgele döndürme (±20°)
   -Yatay ve dikey çevirme (Flip)
   -Rastgele yeniden boyutlandırma ve kırpma (RandomResizedCrop)
   -Parlaklık, kontrast ve doygunlukta rastgele değişiklikler (ColorJitter)
🔹*Veri Bölme:* Veri seti, Eğitim (%70), Doğrulama (%15) ve Test (%15) setlerine ayrılmıştır.

## 2. Basit CNN Mimarisi ve Düzenlileştirme
Dört sınıflı sınıflandırma görevine özel, üç evrişim katmanına sahip basit bir Evrişimsel Sinir Ağı (CNN) mimarisi geliştirilmiştir.

🔹*Evrişim Katmanları:* Sığ bir mimari ile 32 → 64 → 128 filtre çıkışı kullanılarak temel özellik hiyerarşisi öğrenilmiştir.
🔹*Global Havuzlama:* Klasik Flatten yerine AdaptiveAvgPool2d katmanı kullanılarak çıktı özellik vektörü sabit bir boyuta (1×1) getirilmiştir. Bu, modelin mimari esnekliğini artırır.
🔹*Dropout:* Tam bağlantılı katmanlarda Dropout uygulanarak aşırı uydurmanın (overfitting) önüne geçilmiştir.

## 3. Optimizasyon ve Hiperparametre Ayarı
Model performansını en üst düzeye çıkarmak için sistematik bir optimizasyon stratejisi uygulanmıştır.

🔹*Optimizasyon:* Eğitim, Adam optimizasyon algoritması ve L2 düzenlileştirme (Weight Decay) ile gerçekleştirilmiştir.
🔹*Grid Search:* En uygun öğrenme oranı (learning rate) ve Dropout oranı kombinasyonunu bulmak için kapsamlı bir Grid Search uygulanmıştır.
🔹*Kayıp Fonksiyonu:* Çoklu sınıflandırma için standart Cross-Entropy Loss kullanılmıştır.

## 4. Şeffaflık ve Değerlendirme Metotları
Modelin sadece doğruluk oranı değil, aynı zamanda tahmin süreçleri de incelenmiştir.

🔹*Grad-CAM (Gradient-weighted Class Activation Mapping):* Modelin hangi görüntü bölgelerine odaklandığını gösteren ısı haritaları (heatmap) oluşturularak açıklanabilir yapay zeka (XAI) uygulanmıştır. Bu sayede modelin biyolojik olarak anlamlı bölgelere mi yoksa alakasız alanlara mı odaklandığı doğrulanmıştır.
🔹*Kapsamlı Değerlendirme:* Test seti performansı, Karmaşıklık Matrisi (Confusion Matrix) ve detaylı Sınıflandırma Raporu (Precision, Recall, F1-Score) ile analiz edilmiştir.
🔹*Görsel Hata Analizi:* Doğru ve yanlış tahmin edilen örnekler görselleştirilerek modelin hata tipleri detaylıca incelenmiştir.

## 🚀 Elde Edilen Sonuçlar
• Modelin eğitim ve doğrulama süreçleri boyunca kayıp ve doğruluk (loss ve accuracy) grafikleri çizilerek performansı takip edilmiştir.
<img width="925" height="498" alt="kayıpdoğruluk" src="https://github.com/user-attachments/assets/5b72691c-2034-4aa8-8176-950dacac3197" />


• En iyi hiperparametrelerle eğitilen modelin test seti üzerinde performansı karmaşıklık matrisi (confusion matrix) ve sınıflandırma raporu (classification report) ile detaylı olarak değerlendirilmiştir.

<img width="575" height="330" alt="test" src="https://github.com/user-attachments/assets/54cce521-09fb-472a-b01d-6618dba67d54" />
<img width="706" height="576" alt="matris" src="https://github.com/user-attachments/assets/fab1f7d4-b8be-4e65-bb14-5d5a8df65624" />


• Ayrıca, modelin tahminlerinin arkasındaki görsel sebepleri anlamak için Grad-CAM (Gradient-weighted Class Activation Mapping) tekniği uygulanmıştır. Bu teknik, modelin görüntünün hangi bölgelerine odaklandığını bir ısı haritası (heatmap) şeklinde göstererek modelin karar verme sürecini daha şeffaf hale getirir.
<img width="958" height="272" alt="x" src="https://github.com/user-attachments/assets/75d37e68-57fc-4d70-ad33-6533ac81559a" />


• Doğru ve yanlış sınıflandırılan örnekler görselleştirilerek modelin güçlü ve zayıf yönleri incelenmiştir.
<img width="965" height="516" alt="sonuç" src="https://github.com/user-attachments/assets/7d9ab252-5f46-4170-a236-80579a037d02" />

## ✨LİNK
https://www.kaggle.com/code/ceydarkolu/deep-learnining-project
