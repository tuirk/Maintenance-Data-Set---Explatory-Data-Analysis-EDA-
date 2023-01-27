#!/usr/bin/env python
# coding: utf-8

# # Maintenance EDA

# In[1]:


##gerekli kütüphanelerin import edilmesi
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


#veri setinin yüklenmesi
df = pd.read_csv("predictive_maintenance.csv")


# In[3]:


#veri setinin ilk 5 satırının yazdırılması
df.head()


# In[4]:


#veri setinin son 5 satırının yazdırılması
df.tail()


# In[5]:


df.info() 


# In[6]:


df.isnull().sum()
#veri setinde boş alan yoktur


# In[7]:


#kullanılmayacak UDI ve Product ID Satırlarının silinmesi
df.drop(['UDI','Product ID'], axis=1, inplace=True)
df.head()


# In[8]:


#Kelvin olarak verilen sıcaklığın celcius'a çevrimi
df["Air temperature [K]"] = df["Air temperature [K]"] - 272.15
df["Process temperature [K]"] = df["Process temperature [K]"] - 272.15


# In[9]:


# Sütun adlarındaki Kelvinlerin (K) Centigrate(C) olarak değiştirilmesi
df.rename(columns={"Air temperature [K]" : "Air temperature [C]","Process temperature [K]" : "Process temperature [C]"},inplace=True)


# In[10]:


df.head()


# In[11]:


df.info() 


# In[12]:


df.describe()


# In[13]:


#ürün türüne göre dağılım
ax = sns.countplot(x="Type", data=df)


# In[14]:


#sıcaklık farkı isimli yeni bir sütun oluşturulması ve ilk 5 satırın yazdırılması
# temp_diff = Process temperature - Process temperature
df['temp_diff'] = pd.DataFrame(df['Process temperature [C]']-df['Air temperature [C]'])
df.head()


# In[15]:


##indeksin resetlenerek sütun haline getirilmesi
#df.reset_index()


# In[16]:


df.sample(5)


# In[17]:


#veri setindeki tüm değişkenlerin histogram dağılımının yapılması
plt.figure(figsize=(15,10))
for i,col in enumerate(df.columns,1):
    plt.subplot(3,3,i)
    sns.histplot(df[col],kde=True, bins=50)


# In[18]:


import math
import numpy as np
from scipy.stats import shapiro 

import math
import numpy as np
from scipy.stats import kstest
from scipy.stats import lognorm
df_num=df.select_dtypes(["float64","int64"])
for col in df_num:
    print(col)
    plt.figure()
#     ks,p =kstest(df_num[col],'norm')
#     print(ks,p)
    
    shapiro(df_num)
  
    
    


# In[19]:


#Hangi tür arıza tipleri olduğunu görmek için
df['Failure Type'].unique()


# In[20]:


#seçilen tek değişkenin dağılımının basılması
sns.displot(data=df, x="Failure Type", kde=True)


# In[21]:


#seçilen tek değişkenin dağılımının basılması
#(sadece arıza olduğu durumda yani Target=1'ken)
sns.displot(data=df[df['Target'] == 1], x="Failure Type", kde=True)


# In[22]:


#4 tip arıza beklerken 5 tip basılmış, 
#dolayısıyla Arıza türlerinin incelenmesi gerekmektedir
df1=df[df['Target'] == 1]
df1['Failure Type'].value_counts()


# In[23]:


#Target =1 ken yani arıza var ise arıza tipinde no failure olmamalı. Bu veriler yanlış işaretlenmiş
#yanlış işaretlenen verilerin silinmesi
indexNames = df[(df['Target'] == 1) & (df['Failure Type'] == 'No Failure')].index
df.drop(indexNames , inplace=True)


# In[24]:


##yanlış işaretlenen verilerin silinmesinin kontrolü
df2=df[df['Target'] == 1]
df2['Failure Type'].value_counts()


# In[25]:


#seçilen tek değişkenin dağılımının basılması
plt.figure(figsize = (10, 8))
sns.displot(data=df[df['Target'] == 1], x="Failure Type", kde=True)


# In[26]:


##veri setinin arıza türü bazında gruplanması ve değerlerin ortalaması
arizaturu = df.groupby('Failure Type').mean()
arizaturu


# In[27]:


labels_Failed = ["M", "L", "H"]
#türlere göre arızaların ayrılması
M_Failed = sum(df.loc[df['Type']=='M'].Target)
L_Failed = sum(df.loc[df['Type']=='L'].Target)
H_Failed = sum(df.loc[df['Type']=='H'].Target)
Failed=[M_Failed, L_Failed, H_Failed]

#Kalite türlerine göre toplam ürün sayıları
M_Tot = len(df.loc[df['Type']=='M'].Target)
L_Tot = len(df.loc[df['Type']=='L'].Target)
H_Tot = len(df.loc[df['Type']=='H'].Target)

#türlere göre sorunsuz olanların ayrılması
M_NF = M_Tot-M_Failed
L_NF = L_Tot-L_Failed
H_NF = H_Tot-H_Failed

NFail = [M_NF, L_NF, H_NF]

fig, ax = plt.subplots(1,1)
width = 0.3
ax.bar(labels_Failed, Failed, width, label='Arızalı',color='Red')
ax.bar(labels_Failed, NFail, width, bottom=Failed,label='Sağlam',color='green')
ax.set_xlabel('Tür')
ax.set_ylabel('Sayı')
ax.set_title('Üretim Arızaları')
ax.legend()


# In[28]:


#hata yüzdelerinin basılması
print('Sağlam ürünler:',round((M_NF+L_NF+H_NF)*100/(M_Tot+L_Tot+H_Tot),1),'%')
print('Arızalı ürünler:',round((M_Failed+L_Failed+H_Failed)*100/(M_Tot+L_Tot+H_Tot),1),'%')


# In[29]:


#Sıcaklık eksenlerine göre sürü grafiği
plt.figure(figsize=(18,10))
sns.swarmplot(data=df[df['Target'] == 1],x="Process temperature [C]",y='Air temperature [C]',hue="Failure Type")


# In[30]:


#Sürü grafiği farklı kategorik değişkenlerin görselleştirilmesine yardımcı olur. 
#benzer şekilde scatter plot (serpilme grafiği) de 2 tür verinin incelendiği durumda kullanılabilir, 
#bu veri analizi için sürü grafiğine göre daha hızlı sonuç alınmıştır.


# In[31]:


#farklı değişkenlere göre arızaların oluşmasının gözlemlenmesi (arıza olmayan durumlar hariç tutulmadan)
plt.figure(figsize=(18,7))
sns.scatterplot(data=df, x="Process temperature [C]", y="Air temperature [C]", hue="Failure Type");


# In[32]:


#farklı değişkenlere göre arızaların oluşmasının gözlemlenmesi (arızalı olduğu durumda yani df['Target'] == 1 iken)
plt.figure(figsize=(18,7))
sns.scatterplot(data=df[df['Target'] == 1], x="Process temperature [C]", y="Air temperature [C]", hue="Failure Type");


# In[33]:


#bazı durumlarda verilerin belirli bir kısmının analiz edilmesi daha anlaşılır sonuçlar sunabilirken 
#bazılarında tüm verinin değerlendirilmesi daha açıklayıcı olabilir.


# In[34]:


#farklı değişkenlere göre arızaların oluşmasının gözlemlenmesi (arıza olmayan durumlar hariç tutulmadan) -  Torque ve rotational speed için
plt.figure(figsize=(18,7))
sns.scatterplot(data=df, x="Torque [Nm]", y="Rotational speed [rpm]", hue="Failure Type");


# In[35]:


# Torque ve rotational speed değerlerine göre arıza olan durumda (Target = 1 iken) ürün türü dağılımı
plt.figure(figsize=(18,7))
sns.scatterplot(data=df[df['Target'] == 1], x="Torque [Nm]", y="Rotational speed [rpm]", hue="Type");


# In[ ]:




