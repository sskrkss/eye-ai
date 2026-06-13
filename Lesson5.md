# 5. Baseline: от сырых данных до первых метрик

```python
import os
from math import gcd
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from tqdm.auto import tqdm as tqdm_auto
```

## 5.1 Источник и состав данных

В основе проекта — два публичных датасета с соревнований на Kaggle. Каждый содержит снимки глазного дна и разметку к ним:

| Датасет | Год | Снимков train | Снимков test | Разметка train | Разметка test | Формат | Ссылка для скачивания |
|---|---|---|------|---|---------------|---|---|
| DR Detection 2015 | 2015 | 35 126 | 53 576 | ✓ | ✓* | JPEG | [Kaggle](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) |
| APTOS 2019 | 2019 | 3 662 | 1928 | ✓ | —** | PNG | [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) |

*\* Разметку для test DR 2015 можно скачать в [discussion](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/discussion/16149) от организаторов соревнования.*

*\*\* Разметка для test APTOS 2019 организаторами не предоставлялась — снимки исключены из выборки.*

Снимки размечены на 5 классов:

| Класс | Описание                           |
|---|------------------------------------|
| 0 | No DR — норма                      |
| 1 | Mild — легкая                      |
| 2 | Moderate — умеренная               |
| 3 | Severe — тяжелая                   |
| 4 | Proliferative DR — пролиферативная |

---

### Подготовка данных перед запуском ноутбука

**1.** Скачать снимки и поместить в папку `data/raw_images/`:

**2.** Скачать разметку и поместить в папку `data/labels/`. Переименовать в соответствие с таблицей:

| Оригинал                                  | Итоговое имя                 |
|-------------------------------------------|------------------------------|
| `trainLabels.csv` (DR 2015 train)         | `train_2015.csv` |
| `retinopathy_solution.csv` (DR 2015 test) | `test_2015.csv`  |
| `train.csv` (APTOS 2019 train)            | `train_2019.csv` |

**3.** Запустить ячейки ниже — они объединят три файла разметки в единый `data/labels/all.csv`.

```python
test_2015  = pd.read_csv('data/labels/test_2015.csv')
test_2015.head()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>level</th>
      <th>Usage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1_left</td>
      <td>0</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1_right</td>
      <td>0</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2_left</td>
      <td>0</td>
      <td>Public</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2_right</td>
      <td>0</td>
      <td>Public</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3_left</td>
      <td>2</td>
      <td>Private</td>
    </tr>
  </tbody>
</table>
</div>

```python
train_2015 = pd.read_csv('data/labels/train_2015.csv')
train_2015.head()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10_left</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10_right</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13_left</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13_right</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15_left</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
train_2019 = pd.read_csv('data/labels/train_2019.csv')
train_2019.head()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_code</th>
      <th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000c1434d8d7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001639a390f0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0024cdab0c1e</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>002c21358ce6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>005b95c28852</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
test_2015_clean  = test_2015.rename(columns={'image': 'id_code', 'level': 'diagnosis'})[['id_code', 'diagnosis']]
train_2015_clean = train_2015.rename(columns={'image': 'id_code', 'level': 'diagnosis'})[['id_code', 'diagnosis']]
train_2019_clean = train_2019[['id_code', 'diagnosis']]

df = pd.concat([test_2015_clean, train_2015_clean, train_2019_clean], ignore_index=True)
df.to_csv('data/labels/all.csv', index=False)

df.head()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_code</th>
      <th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1_left</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1_right</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2_left</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2_right</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3_left</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

## 5.2 Оценка качества разметки

### 5.2.1 Проверка соответствия снимков и разметки

```python
image_dir = 'data/raw_images'

image_filenames = {p.stem for p in Path(image_dir).iterdir()}
label_filenames = set(df['id_code'])

no_annotation = image_filenames - label_filenames
no_image      = label_filenames - image_filenames

print(f'Снимков на диске:       {len(image_filenames)}')
print(f'Записей в разметке:     {len(label_filenames)}')
print(f'Снимков без разметки:   {len(no_annotation)}')
print(f'Разметка без снимка:    {len(no_image)}')

if no_annotation:
    print('\nСнимки без разметки:', sorted(no_annotation))
if no_image:
    print('\nРазметка без снимков:', sorted(no_image))
```

    Снимков на диске:       92364
    Записей в разметке:     92364
    Снимков без разметки:   0
    Разметка без снимка:    0

**Вывод:** Все снимки на диске имеют соответствующую запись в разметке и наоборот. Датасет полностью согласован.

---

### 5.2.2 Проверка дубликатов

```python
dup_ids = df[df.duplicated(subset='id_code', keep=False)].sort_values('id_code')

exact     = dup_ids[dup_ids.duplicated(subset=['id_code', 'diagnosis'], keep=False)]
conflicts = dup_ids[~dup_ids.duplicated(subset=['id_code', 'diagnosis'], keep=False)]

print(f'Снимков с дублирующейся разметкой: {dup_ids["id_code"].nunique()}')
print(f'  — точные дубликаты (id + разметка совпадают): {exact["id_code"].nunique()}')
print(f'  — конфликтующие метки (один id, разные разметки): {conflicts["id_code"].nunique()}')

if not conflicts.empty:
    print('\nПримеры конфликтов:')
    print(conflicts.head(10).to_string(index=False))
```

    Снимков с дублирующейся разметкой: 0
      — точные дубликаты (id + разметка совпадают): 0
      — конфликтующие метки (один id, разные разметки): 0

**Вывод:** Дубликатов не обнаружено. Разметка технически консистентна.

---

### 5.2.3 Ограничения оценки качества разметки

Проведенные проверки охватывают только **техническую** сторону разметки: полноту, уникальность идентификаторов и согласованность между снимками и CSV. Оценить **медицинскую корректность** разметки — правильность поставленных диагнозов — мы не можем, поскольку это требует профессиональных компетенций врача-офтальмолога. В рамках данного проекта разметка принимается как данность, предоставленная квалифицированными специалистами.

---

## 5.3 EDA

### 5.3.1 Несколько снимков одного пациента

```python
# В DR 2015 id_code выглядит как "10_left" / "10_right" — один пациент, два снимка
dr2015 = df[df['id_code'].str.match(r'.+_(left|right)$')].copy()
dr2015['patient_id'] = dr2015['id_code'].str.rsplit('_', n=1).str[0]

counts = dr2015.groupby('patient_id').size()
both_eyes = counts[counts > 1]

print(f'Пациентов в DR 2015: {dr2015["patient_id"].nunique()}')
print(f'Из них — с снимками левого и правого глаза: {len(both_eyes)} ({len(both_eyes) / dr2015["patient_id"].nunique() * 100:.1f}%)')
```

    Пациентов в DR 2015: 44351
    Из них — с снимками левого и правого глаза: 44351 (100.0%)

**Вывод:** Все пациенты в DR Detection 2015 представлены двумя снимками — левым и правым глазом. Снимки одного пациента визуально похожи: одинаковое качество съемки, схожее состояние сетчатки. Если при разбивке не учитывать это и делить по снимкам, а не по пациентам, один и тот же пациент может попасть одновременно в train и test/validation. Модель будет «видеть» похожие снимки при обучении и при оценке. Это классический пример **утечки данных (data leakage)**, ведущей к переобучению. Чтобы ее исключить, разбивку необходимо делать по пациентам: все снимки одного пациента попадают только в одну из выборок.

---

### 5.3.2 Распределение классов

```python
def find_image(directory, id_code):
    for ext in ('.png', '.jpg', '.jpeg'):
        path = os.path.join(directory, f'{id_code}{ext}')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No image found for {id_code} in {directory}')
```

```python
n_samples = 3
class_labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}

fig, axes = plt.subplots(5, n_samples, figsize=(n_samples * 5, 5 * 5), dpi=100)
for cls in range(5):
    samples = df[df['diagnosis'] == cls].sample(n_samples, random_state=6)
    for col, (_, row) in enumerate(samples.iterrows()):
        img = Image.open(find_image(image_dir, row['id_code'])).convert('RGB')
        img.thumbnail((224, 224), Image.Resampling.LANCZOS)
        ax = axes[cls][col]
        ax.imshow(img, interpolation='lanczos')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if col == 0:
            ax.set_ylabel(class_labels[cls], fontsize=12, fontweight='bold', labelpad=10)

plt.suptitle('Примеры снимков по классам', fontsize=18)
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_15_0.png)
    

```python
counts = df['diagnosis'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    [class_labels[i] for i in counts.index],
    counts.values,
    color=['#4CAF50', '#8BC34A', '#FFC107', '#FF5722', '#D32F2F'],
    edgecolor='black', linewidth=0.5
)

for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f'{count}\n({count/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=10)

ax.set_title('Распределение классов', fontsize=13)
ax.set_xlabel('Класс')
ax.set_ylabel('Количество снимков')
ax.set_ylim(0, counts.max() * 1.18)
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_16_0.png)
    

**Вывод:** Классы распределены неравномерно — на класс No DR приходится 72.7% всех снимков, тогда как доля некоторых других классов опускается ниже 3%. При оценке модели следует выбирать метрики, устойчивые к дисбалансу классов; при разбивке на выборки (train/val/test) — применять стратификацию.

---

### 5.3.3 Разрешение и aspect ratio

```python
sizes = []
for id_code in tqdm_auto(df['id_code'], desc='Чтение размеров'):
    img = Image.open(find_image(image_dir, id_code))
    sizes.append(img.size)

sizes_df = pd.DataFrame(sizes, columns=['width', 'height'])
size_counts = (sizes_df['width'].astype(str) + '×' + sizes_df['height'].astype(str)).value_counts().sort_values(ascending=False)

threshold = 100
main = size_counts[size_counts >= threshold]
other_count = size_counts[size_counts < threshold].sum()
if other_count > 0:
    main = pd.concat([main, pd.Series({'Other': other_count})])

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#888888' if k == 'Other' else 'steelblue' for k in main.index]
bars = ax.bar(main.index, main.values, color=colors, edgecolor='black', linewidth=0.5)

for bar, count in zip(bars, main.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(count), ha='center', va='bottom', fontsize=9)

ax.set_title('Распределение разрешений снимков', fontsize=13)
ax.set_xlabel('Разрешение (ширина×высота)')
ax.set_ylabel('Количество снимков')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_18_1.png)
    

```python
small = sizes_df[(sizes_df['width'] < 224) | (sizes_df['height'] < 224)]
small_counts = (small['width'].astype(str) + '×' + small['height'].astype(str)).value_counts().sort_values(ascending=False)

print(f'Снимков с разрешением меньше 224×224: {len(small)}')
```

    Снимков с разрешением меньше 224×224: 2

```python
def simplify_ratio(w, h):
    d = gcd(w, h)
    return f"{w//d}:{h//d}"

aspect_counts = sizes_df.apply(lambda r: simplify_ratio(r['width'], r['height']), axis=1).value_counts().sort_values(ascending=False)

threshold = 10
main = aspect_counts[aspect_counts >= threshold]
other_count = aspect_counts[aspect_counts < threshold].sum()
if other_count > 0:
    main = pd.concat([main, pd.Series({'Other': other_count})])

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#888888' if k == 'Other' else 'steelblue' for k in main.index]
bars = ax.bar(main.index, main.values, color=colors, edgecolor='black', linewidth=0.5)

for bar, count in zip(bars, main.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(count), ha='center', va='bottom', fontsize=9)

ax.set_title('Распределение aspect ratio снимков', fontsize=13)
ax.set_xlabel('Aspect ratio')
ax.set_ylabel('Количество снимков')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_20_0.png)
    

**Вывод:** Снимки имеют разные разрешения и aspect ratio, большинство — прямоугольные. Для обучения моделей необходимо привести все изображения к единому разрешению 224×224. Чтобы не искажать пропорции, используем паддинг: сначала обрезаем черный фон по контуру глаза, затем дополняем черными полями до квадрата и только потом ресайзим. 2 снимка с разрешением меньше 224×224 будут исключены из выборки.

---

### 5.3.4 Яркость и резкость

```python
path_map = {p.stem: str(p) for p in Path(image_dir).iterdir()}

brightness, blur_scores = [], []

for id_code in tqdm_auto(df['id_code'], desc='Анализ снимков'):
    arr = cv2.imread(path_map[id_code], cv2.IMREAD_GRAYSCALE)
    arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
    mask = arr > 10
    mask_sum = mask.sum()

    brightness.append(arr[mask].mean() if mask_sum > 0 else arr.mean())

    arr_norm = cv2.normalize(arr, np.empty_like(arr), 0.0, 255.0, cv2.NORM_MINMAX).astype(np.uint8)
    lap = cv2.Laplacian(arr_norm, cv2.CV_64F)
    blur_scores.append(lap[mask].var() if mask_sum > 0 else 0)

quality_df = df.copy()
quality_df['brightness'] = brightness
quality_df['blur']       = blur_scores
```

```python
p10 = np.percentile(brightness, 10)
p90 = np.percentile(brightness, 90)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(quality_df['brightness'], bins=60, color='steelblue', edgecolor='black', linewidth=0.4)
axes[0].axvline(p10, color='#D32F2F', linestyle='--', linewidth=1.5, label=f'p10 ({p10:.1f})')
axes[0].axvline(p90, color='#388E3C', linestyle='--', linewidth=1.5, label=f'p90 ({p90:.1f})')
axes[0].set_title('Яркость', fontsize=13)
axes[0].set_xlabel('Средняя яркость FOV')
axes[0].set_ylabel('Количество снимков')
axes[0].legend()

axes[1].hist(quality_df['blur'], bins=400, color='#5C6BC0', edgecolor='black', linewidth=0.4)
axes[1].set_title('Резкость', fontsize=13)
axes[1].set_xlabel('Резкость (Variance of Laplacian)')
axes[1].set_ylabel('Количество снимков')
axes[1].set_xlim(left=0, right=1250)

plt.suptitle('Распределение яркости и резкости снимков', fontsize=14)
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_23_0.png)
    

```python
quality_df[['brightness','blur']].describe(percentiles=[.01,.05,.95,.99])
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>brightness</th>
      <th>blur</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>92364.000000</td>
      <td>92364.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>98.777742</td>
      <td>255.802127</td>
    </tr>
    <tr>
      <th>std</th>
      <td>33.855487</td>
      <td>176.229394</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.908841</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1%</th>
      <td>29.141834</td>
      <td>52.571838</td>
    </tr>
    <tr>
      <th>5%</th>
      <td>46.178820</td>
      <td>81.887204</td>
    </tr>
    <tr>
      <th>95%</th>
      <td>156.312008</td>
      <td>609.383008</td>
    </tr>
    <tr>
      <th>99%</th>
      <td>180.697513</td>
      <td>841.095711</td>
    </tr>
    <tr>
      <th>max</th>
      <td>250.672644</td>
      <td>12595.599708</td>
    </tr>
  </tbody>
</table>
</div>

```python
n = 5
percentile_groups = [
    ('p0.1 (экстремально темные)',  0.1),
    ('p5 (очень темные)',         5),
    ('p30 (темнее нормы)',        30),
    ('p50 (норма)',               50),
    ('p70 (светлее нормы)',       70),
    ('p95 (очень светлые)',       95),
    ('p99.9 (экстремально светлые)', 99.9),
]

fig, axes = plt.subplots(len(percentile_groups), n, figsize=(n * 4, len(percentile_groups) * 4), dpi=100)

for row_idx, (label, pct) in enumerate(percentile_groups):
    target_val = np.percentile(quality_df['brightness'], pct)
    ids = quality_df.iloc[(quality_df['brightness'] - target_val).abs().argsort()[:n]]['id_code'].values
    for col, id_code in enumerate(ids):
        img = Image.open(find_image(image_dir, id_code)).convert('RGB')
        img.thumbnail((224, 224), Image.Resampling.LANCZOS)
        bv = quality_df.loc[quality_df['id_code'] == id_code, 'brightness'].values[0]
        ax = axes[row_idx][col]
        ax.imshow(img, interpolation='lanczos')
        ax.set_title(f'{bv:.1f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[row_idx][0].set_ylabel(label, fontsize=11, fontweight='bold', labelpad=10)

plt.suptitle('Примеры снимков по уровню яркости', fontsize=14)
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_25_0.png)
    

```python
n = 5
p30 = np.percentile(quality_df['brightness'], 30)
p70 = np.percentile(quality_df['brightness'], 70)
normal_exp = quality_df[(quality_df['brightness'] >= p30) & (quality_df['brightness'] <= p70)]

blur_groups = [
    ('p1 (очень размытые)',  normal_exp.nsmallest(n, 'blur')['id_code'].values),
    ('p10 (размытые)',       normal_exp.iloc[(normal_exp['blur'] - np.percentile(normal_exp['blur'], 10)).abs().argsort()[:n]]['id_code'].values),
    ('p50 (средние)',        normal_exp.iloc[(normal_exp['blur'] - np.percentile(normal_exp['blur'], 50)).abs().argsort()[:n]]['id_code'].values),
    ('p90 (резкие)',         normal_exp.iloc[(normal_exp['blur'] - np.percentile(normal_exp['blur'], 90)).abs().argsort()[:n]]['id_code'].values),
]
blur_metric = normal_exp.set_index('id_code')['blur']

fig, axes = plt.subplots(len(blur_groups), n, figsize=(n * 4, len(blur_groups) * 4), dpi=100)

for row_idx, (label, ids) in enumerate(blur_groups):
    for col, id_code in enumerate(ids):
        img = Image.open(find_image(image_dir, id_code)).convert('RGB')
        img.thumbnail((224, 224), Image.Resampling.LANCZOS)
        ax = axes[row_idx][col]
        ax.imshow(img, interpolation='lanczos')
        ax.set_title(f'{blur_metric[id_code]:.1f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[row_idx][0].set_ylabel(label, fontsize=11, fontweight='bold', labelpad=10)

plt.suptitle('Примеры снимков по резкости', fontsize=18)
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_26_0.png)
    

**Вывод:** Измерены две метрики фотометрического качества: **яркость** (средняя интенсивность пикселей FOV) и **резкость** (дисперсия лапласиана). Оба показателя имеют широкий разброс: среди снимков есть экстремально темные (недоэкспонированные) и экстремально светлые (переэкспонированные), а также выражено размытые.

Все это отражает реальную клиническую практику — оборудование и условия съемки варьируются от клиники к клинике. Тем не менее явный брак будет исключен из выборки:

| Условие | Причина                                      |
|---|----------------------------------------------|
| `brightness < 20` | Почти черный снимок, структуры неразличимы   |
| `brightness > 210` | Пересвет, детали в светлых областях потеряны |
| `blur < 30` | Нечитаемо размытый снимок                    |

---

### 5.3.5 Другие артефакты

#### Блики

```python
img = Image.open(find_image(image_dir, '44_right')).convert('RGB')
img.thumbnail((224, 224), Image.Resampling.LANCZOS)
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_29_0.png)
    

---

#### Смещение

```python
img = Image.open(find_image(image_dir, '59ee65760535')).convert('RGB')
img.thumbnail((224, 224), Image.Resampling.LANCZOS)
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_31_0.png)
    

---

#### Пятна на объективе

```python
img = Image.open(find_image(image_dir, '1695_left')).convert('RGB')
img.thumbnail((224, 224), Image.Resampling.LANCZOS)
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_33_0.png)
    

**Вывод:** В датасете присутствуют три типа визуальных артефактов: **блики** (отражение вспышки, маскирующее центральные структуры), **смещение** (камера захватила глазное дно со сдвигом, часть анатомических структур вышла за границу кадра) и **пятна на объективе** (загрязнения линзы, одинаково проявляющиеся на снимках одного устройства). Автоматически обнаружить такие артефакты затруднительно, а ручная фильтрация при объеме в 92 000 снимков нецелесообразна — поэтому все снимки остаются в датасете. В перспективе можно попробовать разработать детектор артефактов и проверить, улучшит ли фильтрация качество модели.

---

## 5.4 Предобработка данных

### 5.4.1 Фильтрация снимков с разрешением меньше 224×224

```python
n_before = len(df)

mask_size = (sizes_df['width'] >= 224) & (sizes_df['height'] >= 224)
df       = df[mask_size].reset_index(drop=True)
sizes_df = sizes_df[mask_size].reset_index(drop=True)

print(f'Снимков до фильтрации:    {n_before}')
print(f'Снимков после фильтрации: {len(df)}')
print(f'Удалено: {n_before - len(df)} снимков с разрешением < 224×224')
```

    Снимков до фильтрации:    92364
    Снимков после фильтрации: 92362
    Удалено: 2 снимков с разрешением < 224×224

### 5.4.2 Фильтрация снимков по яркости и резкости

```python
n_before = len(df)

bad_ids = set(quality_df[
    (quality_df['brightness'] < 20) |
    (quality_df['brightness'] > 210) |
    (quality_df['blur'] < 30)
]['id_code'])
df = df[~df['id_code'].isin(bad_ids)].reset_index(drop=True)

print(f'Снимков до фильтрации:    {n_before}')
print(f'Снимков после фильтрации: {len(df)}')
print(f'Удалено: {n_before - len(df)} снимков по критериям яркости/резкости')
```

    Снимков до фильтрации:    92362
    Снимков после фильтрации: 91953
    Удалено: 409 снимков по критериям яркости/резкости

### 5.4.3 Ресайз снимков до разрешения 224×224 с сохранением aspect ratio

```python
def preprocess(path, desired_size=224, return_steps=False):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Cannot read image: {path}')

    steps = [(f'Оригинал\n{img.shape[1]}×{img.shape[0]}', img.copy())] if return_steps else None

    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f'No contours found (image may be entirely black): {path}')
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    img = img[y:y+h, x:x+w]

    if return_steps:
        steps.append((f'Contour crop\n{img.shape[1]}×{img.shape[0]}', img.copy()))

    h, w = img.shape[:2]
    side = max(h, w)
    padded = np.zeros((side, side, 3), dtype=np.uint8)
    padded[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = img
    img = padded

    if return_steps:
        steps.append((f'Pad до квадрата\n{img.shape[1]}×{img.shape[0]}', img.copy()))

    img = cv2.resize(img, (desired_size, desired_size), interpolation=cv2.INTER_LANCZOS4)

    if return_steps:
        steps.append((f'Resize\n{img.shape[1]}×{img.shape[0]}', img.copy()))
        return img, steps

    return img

sample_path = path_map[df.iloc[0]['id_code']]
_, steps = preprocess(sample_path, return_steps=True)

fig, axes = plt.subplots(1, len(steps), figsize=(len(steps) * 5, 4))
for ax, (title, img) in zip(axes, steps):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.suptitle('Пайплайн предобработки снимка', fontsize=13)
plt.tight_layout()
plt.show()
```

    
![png](data/notebooks/Lesson5/Lesson5_40_0.png)
    

```python
processed_dir = 'data/processed_images'
os.makedirs(processed_dir, exist_ok=True)

for id_code in tqdm_auto(df['id_code'], desc='Предобработка'):
    out_path = os.path.join(processed_dir, f'{id_code}.png')
    if not os.path.exists(out_path):
        img = preprocess(path_map[id_code])
        cv2.imwrite(out_path, img)

processed_paths = list(Path(processed_dir).glob('*.png'))
print(f'Снимков в папке:   {len(processed_paths)}')
print(f'Снимков в разметке: {len(df)}')

wrong_size = [p.name for p in processed_paths if cv2.imread(str(p)).shape[:2] != (224, 224)]
if wrong_size:
    print(f'\nСнимков с неверным разрешением: {len(wrong_size)}')
    print(wrong_size[:10])
else:
    print(f'\nВсе снимки имеют разрешение {224}×{224} ✓')
```

    Снимков в папке:   91953
    Снимков в разметке: 91953
    
    Все снимки имеют разрешение 224×224 ✓

### 5.4.4 Разбиение датасета на train/test

Датасет разбивается в соотношении **80/20** по пациентам — все снимки одного пациента попадают только в одну из выборок, чтобы исключить утечку данных. Стратификация по `diagnosis` сохраняется в рамках группового разбиения. Тестовая выборка используется для финальной оценки качества обученных моделей.

```python
# Для DR 2015: "10_left" → patient_id="10". Для APTOS 2019: хэш без суффикса → остаётся как есть
df['patient_id'] = df['id_code'].str.rsplit('_', n=1).apply(
    lambda parts: parts[0] if len(parts) > 1 and parts[1] in ('left', 'right') else '_'.join(parts)
)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, df['diagnosis'], groups=df['patient_id']))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

train_df.to_csv('data/labels/train.csv', index=False)
test_df.to_csv('data/labels/test.csv', index=False)

print(f'Train: {len(train_df)} снимков ({train_df["patient_id"].nunique()} пациентов)')
print(f'Test:  {len(test_df)} снимков ({test_df["patient_id"].nunique()} пациентов)\n')

train_counts = train_df['diagnosis'].value_counts().sort_index()
test_counts  = test_df['diagnosis'].value_counts().sort_index()

print(pd.DataFrame({
    'train':   train_counts,
    'train %': (train_counts / len(train_df) * 100).round(1),
    'test':    test_counts,
    'test %':  (test_counts  / len(test_df)  * 100).round(1),
}).rename_axis('class'))
```

    Train: 73550 снимков (38364 пациентов)
    Test:  18403 снимков (9591 пациентов)
    
           train  train %   test  test %
    class                               
    0      53509     72.8  13365    72.6
    1       5198      7.1   1362     7.4
    2      11261     15.3   2815    15.3
    3       1828      2.5    429     2.3
    4       1754      2.4    432     2.3

### 5.4.5 Разбиение train на фолды (кросс-валидация)

Тренировочная выборка разбивается на **5 фолдов** по пациентам — пациенты из одного фолда не пересекаются с другими. Стратификация по `diagnosis` сохраняет распределение классов в каждом фолде. Каждая итерация кросс-валидации использует 4 фолда для обучения модели и 1 для оценки ее качества.

*Примечание:* в дальнейшем данный подход может быть пересмотрен, вплоть до использования всей тренировочной выборки для обучения финальной модели.

```python
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

train_df = train_df.reset_index(drop=True)
train_df['fold'] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_df, train_df['diagnosis'], groups=train_df['patient_id'])):
    train_df.loc[val_idx, 'fold'] = fold

train_df.to_csv('data/labels/train.csv', index=False)

fold_sizes = train_df.groupby('fold').size().rename('count')
print(fold_sizes.to_string())
print(f'\nИтого: {fold_sizes.sum()} снимков, {len(fold_sizes)} фолдов')
```

    fold
    0    14710
    1    14710
    2    14710
    3    14710
    4    14710
    
    Итого: 73550 снимков, 5 фолдов

```python
train_df.head()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_code</th>
      <th>diagnosis</th>
      <th>patient_id</th>
      <th>fold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1_left</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1_right</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2_left</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2_right</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4_left</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

### 5.4.6 Версионирование данных с DVC

Обработанные снимки и разметка добавляются в DVC

```python
!dvc add data/labels/train.csv data/labels/test.csv
```

```python
!dvc add data/processed_images
```

```python
!dvc push
```

## 5.5 Обучение baseline моделей

Для обучения baseline моделей выбраны шесть архитектур:

| Модель | Год | Тип архитектуры | Размороженные слои | Всего параметров | Обучаемых параметров |
|---|---|---|---|---|---|
| VGG-19 | 2014 | CNN | `classifier` + `features[28:]` | ~144M | ~124M |
| ResNet-50 | 2015 | CNN | `fc` + `layer4` | ~26M | ~15M |
| EfficientNetV2S | 2022 | CNN | `classifier` + `features[-2:]` | ~21M | ~5M |
| DenseNet-201 | 2017 | CNN | `classifier` + `denseblock4` + `norm5` | ~20M | ~8M |
| Swin-V2-Tiny | 2022 | Vision Transformer (shifted windows) | `head` + `features[-1]` + `norm` | ~28M | ~14M |
| ViT-B/16 | 2020 | Vision Transformer | `heads` + `encoder.layers[-1]` + `encoder.ln` | ~86M | ~7M |

Все модели предобучены на ImageNet. Чтобы адаптировать их к задаче (5 классов вместо 1000), заменяется последний слой, затем размораживаются новый классификатор и последний блок backbone — более глубокие слои остаются замороженными.

Параметры обучения:

| Параметр       | Значение                                                           |
|----------------|--------------------------------------------------------------------|
| Оптимизатор    | Adam с дифференциальными LR:<br/>1e-3 (классификатор) / 1e-4 (backbone) |
| Лосс           | CrossEntropyLoss                                                   |
| Батч           | 32                                                                 |
| Максимум эпох  | 20                                                                 |
| Early stopping | 3 эпохи                                                            |
| Аугментация    | случайное горизонтальное отражение<br>случайный поворот до 180°<br>случайный сдвиг, масштабирование (±20%) и shear 11.5°<br>случайное изменение яркости, контраста, насыщенности и оттенка<br>размытие по Гауссу с вероятностью 0.5 |

*Early stopping*: обучение прерывается, если test loss не улучшается 3 эпохи подряд; восстанавливаются веса лучшей эпохи.

*Аугментация*: применяется только к тренировочной выборке.


```python
import copy
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import (
    vgg19, resnet50, swin_v2_t, efficientnet_v2_s, densenet201, vit_b_16,
    VGG19_Weights, ResNet50_Weights, Swin_V2_T_Weights,
    EfficientNet_V2_S_Weights, DenseNet201_Weights, ViT_B_16_Weights,
)
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# Дублируем некоторые функции, чтобы можно было запускать ячейки с этого места
def find_image(directory, id_code):
    for ext in ('.png', '.jpg', '.jpeg'):
        path = os.path.join(directory, f'{id_code}{ext}')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No image found for {id_code} in {directory}')


class DRDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.records = df[['id_code', 'diagnosis']].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img = read_image(find_image(self.image_dir, row['id_code']))
        if self.transform:
            img = self.transform(img)
        return img, int(row['diagnosis'])
```


```python
def make_train_transform(weights):
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=11.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.2, saturation=0.08, hue=0.028),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    ])
    return transforms.Compose([aug, weights.transforms()])


def train_epoch(model, optimizer, loader, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc='Train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    for x, y in tqdm(loader, desc='Eval', leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item()
    return total_loss / len(loader)


def plot_stats(train_loss, test_loss, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f'{title} — Loss')
    ax.plot(train_loss, label='Train')
    ax.plot(test_loss, label='Test')
    ax.legend(); ax.grid()
    plt.tight_layout(); plt.show()


def fit(model, optimizer, train_loader, test_loader, max_epochs, title, loss_fn, early_stopping):
    train_losses, test_losses = [], []

    best_test_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        tr_loss = train_epoch(model, optimizer, train_loader, loss_fn)
        ts_loss = evaluate(model, test_loader, loss_fn)
        train_losses.append(tr_loss)
        test_losses.append(ts_loss)
        clear_output(wait=True)
        plot_stats(train_losses, test_losses, title)
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss {tr_loss:.4f} | Test Loss {ts_loss:.4f}")

        if ts_loss < best_test_loss:
            best_test_loss = ts_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping:
                print(f"Early stopping на эпохе {epoch+1} (test loss не улучшался {early_stopping} эпох)")
                break

    model.load_state_dict(best_model_state)


def train_model(
        name,
        weights,
        model,
        optimizer,
        loss_fn,
        train_df_path,
        test_df_path,
        images_dir,
        models_dir,
        max_epochs,
        batch_size,
        early_stopping,
):
    train_df = pd.read_csv(train_df_path)
    test_df  = pd.read_csv(test_df_path)

    fname = name.lower().replace('-', '').replace(' ', '_')

    train_transform = make_train_transform(weights)
    test_transform  = weights.transforms()

    train_loader = DataLoader(DRDataset(train_df, images_dir, train_transform), batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(DRDataset(test_df,  images_dir, test_transform),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n{'='*40}\n{name}\n{'='*40}")
    fit(model, optimizer, train_loader, test_loader, max_epochs=max_epochs, title=name, loss_fn=loss_fn, early_stopping=early_stopping)

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f'{fname}.pth')
    torch.save(model.state_dict(), path)
    print(f"Сохранено: {path}")
```


```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
print(f"Device: {device}")
```

    Device: cuda:0



```python
train_df_path = 'data/labels/train.csv'
test_df_path = 'data/labels/test.csv'
images_dir = 'data/processed_images'
models_dir = 'data/models/baseline'
max_epochs    = 20
batch_size    = 32
early_stopping = 5
```


### 5.5.1 VGG-19


```python
vgg_weights = VGG19_Weights.DEFAULT
model_vgg = vgg19(weights=vgg_weights)

for p in model_vgg.parameters():
    p.requires_grad = False

model_vgg.classifier[6] = nn.Linear(4096, 5)

model_vgg.classifier.requires_grad_(True)
for i in range(28, len(model_vgg.features)):
    for p in model_vgg.features[i].parameters():
        p.requires_grad = True

model_vgg = model_vgg.to(device)

optimizer_vgg = Adam([
    {'params': model_vgg.classifier.parameters(), 'lr': 1e-3},
    {'params': [p for i in range(28, len(model_vgg.features))
                  for p in model_vgg.features[i].parameters()], 'lr': 1e-4},
])

train_model(
    name='VGG-19',
    weights=vgg_weights,
    model=model_vgg,
    optimizer=optimizer_vgg,
    loss_fn=loss_fn,
    train_df_path=train_df_path,
    test_df_path=test_df_path,
    images_dir=images_dir,
    models_dir=models_dir,
    max_epochs=max_epochs,
    batch_size=batch_size,
    early_stopping=early_stopping,
)
```


    
![png](data/notebooks/Lesson5/Lesson5_60_0.png)
    


    Epoch 20/20 | Train Loss 0.7244 | Test Loss 0.7197
    Сохранено: /content/vgg19.pth


### 5.5.2 ResNet-50


```python
resnet_weights = ResNet50_Weights.DEFAULT
model_resnet = resnet50(weights=resnet_weights)

for p in model_resnet.parameters():
    p.requires_grad = False

model_resnet.fc = nn.Linear(2048, 5)

model_resnet.fc.requires_grad_(True)
model_resnet.layer4.requires_grad_(True)

model_resnet = model_resnet.to(device)

optimizer_resnet = Adam([
    {'params': model_resnet.fc.parameters(),     'lr': 1e-3},
    {'params': model_resnet.layer4.parameters(), 'lr': 1e-4},
])

train_model(
    name='ResNet-50',
    weights=resnet_weights,
    model=model_resnet,
    optimizer=optimizer_resnet,
    loss_fn=loss_fn,
    train_df_path=train_df_path,
    test_df_path=test_df_path,
    images_dir=images_dir,
    models_dir=models_dir,
    max_epochs=max_epochs,
    batch_size=batch_size,
    early_stopping=early_stopping,
)
```


    
![png](data/notebooks/Lesson5/Lesson5_62_0.png)
    


    Epoch 20/20 | Train Loss 0.6179 | Test Loss 0.6422
    Сохранено: /content/resnet50.pth


### 5.5.3 EfficientNetV2S


```python
eff_weights = EfficientNet_V2_S_Weights.DEFAULT
model_eff = efficientnet_v2_s(weights=eff_weights)

for p in model_eff.parameters():
    p.requires_grad = False

model_eff.classifier[1] = nn.Linear(1280, 5)

model_eff.classifier.requires_grad_(True)
model_eff.features[-1].requires_grad_(True)
model_eff.features[-2].requires_grad_(True)

model_eff = model_eff.to(device)

optimizer_eff = Adam([
    {'params': model_eff.classifier.parameters(), 'lr': 1e-3},
    {'params': list(model_eff.features[-1].parameters()) +
               list(model_eff.features[-2].parameters()), 'lr': 1e-4},
])

train_model(
    name='EfficientNetV2S',
    weights=eff_weights,
    model=model_eff,
    optimizer=optimizer_eff,
    loss_fn=loss_fn,
    train_df_path=train_df_path,
    test_df_path=test_df_path,
    images_dir=images_dir,
    models_dir=models_dir,
    max_epochs=max_epochs,
    batch_size=batch_size,
    early_stopping=early_stopping,
)
```


    
![png](data/notebooks/Lesson5/Lesson5_64_0.png)
    


    Epoch 19/20 | Train Loss 0.5623 | Test Loss 0.5845
    Early stopping на эпохе 19 (test loss не улучшался 5 эпох)
    Сохранено: /content/efficientnetv2s.pth


### 5.5.4 DenseNet-201


```python
dens_weights = DenseNet201_Weights.DEFAULT
model_dens = densenet201(weights=dens_weights)

for p in model_dens.parameters():
    p.requires_grad = False

model_dens.classifier = nn.Linear(1920, 5)

model_dens.classifier.requires_grad_(True)
model_dens.features.denseblock4.requires_grad_(True)
model_dens.features.norm5.requires_grad_(True)

model_dens = model_dens.to(device)

optimizer_dens = Adam([
    {'params': model_dens.classifier.parameters(), 'lr': 1e-3},
    {'params': list(model_dens.features.denseblock4.parameters()) +
               list(model_dens.features.norm5.parameters()), 'lr': 1e-4},
])

train_model(
    name='DenseNet-201',
    weights=dens_weights,
    model=model_dens,
    optimizer=optimizer_dens,
    loss_fn=loss_fn,
    train_df_path=train_df_path,
    test_df_path=test_df_path,
    images_dir=images_dir,
    models_dir=models_dir,
    max_epochs=max_epochs,
    batch_size=batch_size,
    early_stopping=early_stopping,
)
```


    
![png](data/notebooks/Lesson5/Lesson5_66_0.png)
    


    Epoch 16/20 | Train Loss 0.6624 | Test Loss 0.7595
    Early stopping на эпохе 16 (test loss не улучшался 5 эпох)
    Сохранено: /content/densenet201.pth


### 5.5.5 Swin-V2-Tiny


```python
swin_weights = Swin_V2_T_Weights.DEFAULT
model_swin = swin_v2_t(weights=swin_weights)

for p in model_swin.parameters():
    p.requires_grad = False

model_swin.head = nn.Linear(768, 5)

model_swin.head.requires_grad_(True)
model_swin.features[-1].requires_grad_(True)
model_swin.norm.requires_grad_(True)

model_swin = model_swin.to(device)

optimizer_swin = Adam([
    {'params': model_swin.head.parameters(), 'lr': 1e-3},
    {'params': list(model_swin.features[-1].parameters()) +
               list(model_swin.norm.parameters()), 'lr': 1e-4},
])

train_model(
    name='Swin-V2-T',
    weights=swin_weights,
    model=model_swin,
    optimizer=optimizer_swin,
    loss_fn=loss_fn,
    train_df_path=train_df_path,
    test_df_path=test_df_path,
    images_dir=images_dir,
    models_dir=models_dir,
    max_epochs=max_epochs,
    batch_size=batch_size,
    early_stopping=early_stopping,
)
```


    
![png](data/notebooks/Lesson5/Lesson5_68_0.png)
    


    Epoch 20/20 | Train Loss 0.6484 | Test Loss 0.6311
    Сохранено: /content/swinv2t.pth


### 5.5.6 ViT-B/16


```python
vit_weights = ViT_B_16_Weights.DEFAULT
model_vit = vit_b_16(weights=vit_weights)

for p in model_vit.parameters():
    p.requires_grad = False

model_vit.heads.head = nn.Linear(768, 5)

model_vit.heads.requires_grad_(True)
model_vit.encoder.layers[-1].requires_grad_(True)
model_vit.encoder.ln.requires_grad_(True)

model_vit = model_vit.to(device)

optimizer_vit = Adam([
    {'params': model_vit.heads.parameters(), 'lr': 1e-3},
    {'params': list(model_vit.encoder.layers[-1].parameters()) +
               list(model_vit.encoder.ln.parameters()), 'lr': 1e-4},
])

train_model(
    name='ViT-B/16',
    weights=vit_weights,
    model=model_vit,
    optimizer=optimizer_vit,
    loss_fn=loss_fn,
    train_df_path=train_df_path,
    test_df_path=test_df_path,
    images_dir=images_dir,
    models_dir=models_dir,
    max_epochs=max_epochs,
    batch_size=batch_size,
    early_stopping=early_stopping,
)
```


    
![png](data/notebooks/Lesson5/Lesson5_70_0.png)
    


    Epoch 15/20 | Train Loss 0.6347 | Test Loss 0.6673
    Early stopping на эпохе 15 (test loss не улучшался 5 эпох)
    Сохранено: /content/vitb16.pth


### 5.5.7 Версионирование baseline моделей с DVC

Обученные чекпоинты добавляются в DVC


```python
!dvc add models/
```


```python
!dvc push
```

## 5.6 Оценка качества baseline моделей

Для оценки качества моделей выбраны **7 метрик**: 5 многоклассовых, учитывающих ординальность и дисбаланс классов, и 2 бинарные, клинически значимые.

**Многоклассовые метрики:**

| Метрика | Описание |
|---|---|
| **QWK** (Quadratic Weighted Kappa) | Взвешенная мера согласия между предсказаниями и разметкой; ошибки на далеких классах штрафуются сильнее, чем на соседних |
| **AUC** (macro OvR) | Площадь под ROC-кривой в схеме one-vs-rest; не зависит от порога классификации, усредняется по классам |
| **Precision** (macro) | Доля верно предсказанных примеров среди всех предсказанных для данного класса; macro-среднее по классам |
| **Recall** (macro) | Доля верно найденных примеров среди всех реальных примеров класса; macro-среднее по классам |
| **F1** (macro) | Гармоническое среднее Precision и Recall; macro-среднее по классам |

**Бинарные метрики — Sensitivity и Specificity** вычисляются после сворачивания 5-классовой шкалы DR в бинарную по одному из трех клинических порогов:

| Порог | Positive (больной) | Negative (здоровый) | Клинический смысл |
|---|---|---|---|
| **Any DR** | классы 1–4 | класс 0 | Есть ли заболевание? |
| **RDR** (Referable DR) | классы 2–4 | классы 0–1 | Нужно ли направить к офтальмологу? |
| **STDR** (Sight-Threatening DR) | классы 3–4 | классы 0–2 | Есть ли угроза потери зрения? |

- **Sensitivity** = TP / (TP + FN) — доля больных, которых модель смогла распознать
- **Specificity** = TN / (TN + FP) — доля здоровых, которых модель не отправила на обследование по ошибке


```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.models import (
    vgg19, resnet50, swin_v2_t, efficientnet_v2_s, densenet201, vit_b_16,
    VGG19_Weights, ResNet50_Weights, Swin_V2_T_Weights,
    EfficientNet_V2_S_Weights, DenseNet201_Weights, ViT_B_16_Weights,
)
from tqdm import tqdm
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
```


```python
# Дублируем некоторые функции, чтобы можно было запускать ячейки с этого места
def find_image(directory, id_code):
    for ext in ('.png', '.jpg', '.jpeg'):
        path = os.path.join(directory, f'{id_code}{ext}')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No image found for {id_code} in {directory}')


class DRDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.records = df[['id_code', 'diagnosis']].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img = read_image(find_image(self.image_dir, row['id_code']))
        if self.transform:
            img = self.transform(img)
        return img, int(row['diagnosis'])
```


```python
def _make_vgg19():
    w = VGG19_Weights.DEFAULT
    m = vgg19(weights=None)
    m.classifier[6] = nn.Linear(4096, 5)
    return m, w

def _make_resnet50():
    w = ResNet50_Weights.DEFAULT
    m = resnet50(weights=None)
    m.fc = nn.Linear(2048, 5)
    return m, w

def _make_efficientnetv2s():
    w = EfficientNet_V2_S_Weights.DEFAULT
    m = efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(1280, 5)
    return m, w

def _make_densenet201():
    w = DenseNet201_Weights.DEFAULT
    m = densenet201(weights=None)
    m.classifier = nn.Linear(1920, 5)
    return m, w

def _make_swin_v2_t():
    w = Swin_V2_T_Weights.DEFAULT
    m = swin_v2_t(weights=None)
    m.head = nn.Linear(768, 5)
    return m, w

def _make_vit_b_16():
    w = ViT_B_16_Weights.DEFAULT
    m = vit_b_16(weights=None)
    m.heads.head = nn.Linear(768, 5)
    return m, w

model_registry = [
    ('VGG-19',          'vgg19',          _make_vgg19),
    ('ResNet-50',       'resnet50',        _make_resnet50),
    ('EfficientNetV2S', 'efficientnetv2s', _make_efficientnetv2s),
    ('DenseNet-201',    'densenet201',     _make_densenet201),
    ('Swin-V2-T',       'swinv2t',         _make_swin_v2_t),
    ('ViT-B/16',        'vitb16',          _make_vit_b_16),
]
```


```python
train_df_path = 'data/labels/train.csv'
test_df_path = 'data/labels/test.csv'
images_dir = 'data/processed_images'
models_dir = 'data/models/baseline'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
```

    Device: cuda:0



```python
@torch.inference_mode()
def get_predictions(model, loader, desc='Inference'):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for x, y in tqdm(loader, desc=desc, leave=True):
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.numpy())
    probs_out  = np.concatenate(all_probs)
    preds_out  = np.concatenate(all_preds)
    labels_out = np.concatenate(all_labels)
    print(f'  [{desc}] samples processed: {len(preds_out)}')
    return probs_out, preds_out, labels_out


def compute_metrics(y_true, y_pred, y_probs):
    return {
        'QWK':       cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'AUC':       roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro'),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall':    recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1':        f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
```


```python
test_df_eval = pd.read_csv(test_df_path)
print(f'Test size: {len(test_df_eval)}\n')

results = {}

for name, fname, model_fn in model_registry:
    print(f"\n{'='*40}\n{name}\n{'='*40}")

    model, weights = model_fn()
    ckpt_path = os.path.join(models_dir, f'{fname}.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    test_transform = weights.transforms()
    test_loader = DataLoader(DRDataset(test_df_eval, images_dir, test_transform), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    test_probs, test_preds, test_labels = get_predictions(model, test_loader, desc=f'{name} test')

    results[name] = {
        'test':        compute_metrics(test_labels, test_preds, test_probs),
        'test_preds':  test_preds,
        'test_labels': test_labels,
    }

    print(f"  QWK  test={results[name]['test']['QWK']:.4f}")
    print(f"  AUC  test={results[name]['test']['AUC']:.4f}")
```

    Test size: 18403
    
    
    ========================================
    VGG-19
    ========================================


    VGG-19 test: 100%|██████████| 288/288 [00:21<00:00, 13.32it/s]


      [VGG-19 test] samples processed: 18403
      QWK  test=0.5339
      AUC  test=0.7926
    
    ========================================
    ResNet-50
    ========================================


    ResNet-50 test: 100%|██████████| 288/288 [00:19<00:00, 14.79it/s]


      [ResNet-50 test] samples processed: 18403
      QWK  test=0.6318
      AUC  test=0.8387
    
    ========================================
    EfficientNetV2S
    ========================================


    EfficientNetV2S test: 100%|██████████| 288/288 [00:36<00:00,  7.86it/s]


      [EfficientNetV2S test] samples processed: 18403
      QWK  test=0.7092
      AUC  test=0.8605
    
    ========================================
    DenseNet-201
    ========================================


    DenseNet-201 test: 100%|██████████| 288/288 [00:21<00:00, 13.64it/s]


      [DenseNet-201 test] samples processed: 18403
      QWK  test=0.5718
      AUC  test=0.8286
    
    ========================================
    Swin-V2-T
    ========================================


    Swin-V2-T test: 100%|██████████| 288/288 [00:25<00:00, 11.10it/s]


      [Swin-V2-T test] samples processed: 18403
      QWK  test=0.6234
      AUC  test=0.8381
    
    ========================================
    ViT-B/16
    ========================================


    ViT-B/16 test: 100%|██████████| 288/288 [00:39<00:00,  7.37it/s]

      [ViT-B/16 test] samples processed: 18403
      QWK  test=0.6146
      AUC  test=0.8315


    



```python
metric_cols = ['QWK', 'AUC', 'Precision', 'Recall', 'F1']

rows = []
for name in results:
    row = {'Model': name}
    row.update(results[name]['test'])
    rows.append(row)

comparison_df = pd.DataFrame(rows).set_index('Model')[metric_cols]

cell_text = [[f'{v:.4f}' for v in row] for row in comparison_df.values]

fig, ax = plt.subplots(figsize=(10, 1.8))
ax.axis('off')

table = ax.table(
    cellText=cell_text,
    rowLabels=comparison_df.index.tolist(),
    colLabels=metric_cols,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for j, col in enumerate(metric_cols):
    best_val = comparison_df[col].max()
    for i, val in enumerate(comparison_df[col]):
        if val == best_val:
            table[i + 1, j].set_facecolor('#d4edda')

plt.tight_layout()
plt.show()
```


    
![png](data/notebooks/Lesson5/Lesson5_81_0.png)
    



```python
class_labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}

n_models = len(model_registry)
ncols = 2
nrows = (n_models + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6))
axes = axes.flatten()

for ax, (name, _, _) in zip(axes, model_registry):
    cm = confusion_matrix(results[name]['test_labels'], results[name]['test_preds'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[class_labels[i] for i in range(5)])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=13)
    ax.tick_params(axis='x', rotation=45)

for ax in axes[n_models:]:
    ax.set_visible(False)

plt.suptitle('Confusion Matrix — Test set', fontsize=14)
plt.tight_layout()
plt.show()
```


    
![png](data/notebooks/Lesson5/Lesson5_82_0.png)
    



```python
thresholds = {
    'Any DR':  1,   # positive: классы 1–4
    'RDR':     2,   # positive: классы 2–4 (referable)
    'STDR':    3,   # positive: классы 3–4 (sight-threatening)
}

rows = []
for name, _, _ in model_registry:
    y_true = results[name]['test_labels']
    y_pred = results[name]['test_preds']
    row = {'Model': name}
    for thresh_name, thresh in thresholds.items():
        y_true_bin = (y_true >= thresh).astype(int)
        y_pred_bin = (y_pred >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        row[f'{thresh_name} Sens'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        row[f'{thresh_name} Spec'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rows.append(row)

df_thresh = pd.DataFrame(rows).set_index('Model')

col_labels = list(df_thresh.columns)
cell_text  = [[f'{v:.4f}' for v in row] for row in df_thresh.values]

fig, ax = plt.subplots(figsize=(14, 2.2))
ax.axis('off')
table = ax.table(
    cellText=cell_text,
    rowLabels=df_thresh.index.tolist(),
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.7)

# Подсвечиваем лучший результат в каждой колонке
for j, col in enumerate(col_labels):
    best = df_thresh[col].max()
    for i, val in enumerate(df_thresh[col]):
        if val == best:
            table[i + 1, j].set_facecolor('#d4edda')

plt.title('Sensitivity / Specificity по клиническим порогам (Test set)', fontsize=12, pad=10)
plt.tight_layout()
plt.show()
```


    
![png](data/notebooks/Lesson5/Lesson5_83_0.png)
    


**Вывод:**

Для дальнейшей оптимизации выбраны по одной лучшей модели из каждой архитектурной группы:
- **EfficientNetV2S** (CNN) — наилучший результат среди всех baseline: QWK=0.71, AUC=0.86
- **Swin-V2-T** (ViT) — лучший среди трансформеров: QWK=0.62, AUC=0.84

Во всех моделях наблюдается резкий дисбаланс: Specificity стабильно высокая (0.96–0.99), тогда как Sensitivity существенно ниже (0.35–0.65). Это артефакт дисбаланса классов: датасет преимущественно состоит из класса 0 (No DR), поэтому модели «по умолчанию» легко правильно классифицируют здоровых (высокий TN), но пропускают значительную долю больных (высокий FN).

Для скрининга это критично: пропуск пациента с заболеванием (низкая Sensitivity) клинически опаснее, чем ложная тревога. EfficientNetV2S показывает наилучшую Sensitivity по всем трем клиническим порогам (Any DR: 0.51, RDR: 0.65, STDR: 0.59), что делает его приоритетным кандидатом. Swin-V2-T уступает по Sensitivity (Any DR: 0.40, RDR: 0.51, STDR: 0.43), однако сохраняет потенциал: трансформеры, как правило, сильнее реагируют на аугментацию и балансировку классов.

Дальнейшая оптимизация будет направлена прежде всего на повышение Sensitivity при сохранении приемлемой Specificity.

---

## 5.7 Направления для оптимизации

- **Более агрессивная фильтрация по яркости и резкости**: текущие пороги (`brightness < 20`, `brightness > 210`, `blur < 30`) отсекают только явный брак. Ужесточение порогов позволит убрать снимки низкого качества, которые затрудняют обучение и могут ухудшать метрики.
- **Undersampling доминирующего класса**: класс No DR составляет ~73% выборки. Уменьшение его доли до уровня остальных классов может помочь модели лучше распознавать редкие классы.
- **Оптимизация аугментации**: текущий набор подобран без экспериментов. Поскольку val loss ниже train loss, аугментации могут быть избыточно агрессивными — их смягчение стоит рассмотреть в первую очередь.
- **Более глубокое размораживание backbone**: сейчас размораживается только последний блок каждой модели. Постепенное размораживание дополнительных слоев может позволить модели адаптировать больше признаков под специфику снимков глазного дна.
- **Loss с весами классов**: присвоить редким классам больший вес в функции потерь, чтобы ошибки на них штрафовались сильнее.
- **SmoothL1Loss**: переформулировать задачу как регрессию на метки 0–4 и округлять предсказание до ближайшего класса. В отличие от CrossEntropyLoss, штрафует дальние ошибки сильнее близких — что соответствует ординальной природе DR.
- **Подбор гиперпараметров**: learning rate, batch size и scheduler выставлены вручную без оптимизации. Автоматический подбор — один из наиболее очевидных резервов роста качества.

---

