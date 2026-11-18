# Baguette magique : explication du code

Le [notebook](src/model_train.ipynb) entraîne un modèle à reconnaître quatre classes : `circle`, `hor`, `nothing` et `ver`. Le programme [main.cpp](src/main.cpp) utilise ce modèle sur une ESP32-S3-Zero, à partir d’un MPU6050, puis affiche le résultat sur un écran QAPASS 1602A.

## Le notebook d’entraînement

Exécuter les cellules dans l’ordre : elles réutilisent les variables créées dans les cellules précédentes.

### Charger les CSV et attribuer les classes

```python
for label, name in enumerate(class_names):
    for file in sorted((dataset / name).glob("*.csv")):
        sequences.append(np.loadtxt(file, delimiter=",", skiprows=1, usecols=(0, 1, 3, 4), dtype=np.float32))
        labels.append(label)
```

`class_names` contient les noms des sous-dossiers, triés alphabétiquement. `enumerate` associe un numéro à chaque classe : `circle = 0`, `hor = 1`, `nothing = 2`, `ver = 3`.

`np.loadtxt` lit un CSV. `skiprows=1` ignore les titres des colonnes et `usecols=(0, 1, 3, 4)` conserve uniquement **X, Y, GX et GY**. Chaque fichier est une séquence complète de 75 mesures, déjà normalisées.

```python
X = np.stack(sequences)
y = np.array(labels)
```

`X` rassemble les 800 gestes dans un tableau de dimensions **(800, 75, 4)** : gestes, mesures, axes. `y` contient les 800 numéros de classe correspondants.

### Afficher un exemple et séparer les données

```python
example = X[y == label][0]
ax.plot(example)
```

`y == label` sélectionne les gestes d’une classe. `[0]` prend le premier exemple et `plot` trace ses quatre axes au fil des mesures.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

`test_size=0.2` réserve 20 % des séquences pour le test : **640 gestes pour apprendre, 160 pour évaluer**. `stratify=y` conserve les proportions des classes. `random_state=42` permet de retrouver la même séparation à chaque exécution.

### Préparer les entrées du CNN

```python
train_x = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
train_y = torch.tensor(y_train, dtype=torch.long)
loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
```

`torch.tensor` convertit les tableaux NumPy en tenseurs PyTorch. `transpose(1, 2)` passe de `(640, 75, 4)` à **(640, 4, 75)**, car `Conv1d` attend les axes avant les mesures temporelles.

`TensorDataset` associe chaque geste à sa classe. `DataLoader` fournit des lots de 32 gestes et mélange leur ordre à chaque époque avec `shuffle=True`.

### Construire le réseau

```python
model = nn.Sequential(
    nn.Conv1d(4, 16, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),
    nn.Conv1d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(4),
    nn.Flatten(),
    nn.Linear(32 * 4, len(class_names)),
)
```

`nn.Sequential` fait traverser les couches dans l’ordre. La première convolution apprend 16 filtres à partir des quatre axes, en observant cinq mesures à la fois. La seconde apprend 32 filtres sur les résultats précédents, avec des fenêtres de trois mesures.

`ReLU` remplace les valeurs négatives par zéro. `MaxPool1d(2)` réduit la longueur temporelle de 75 à 37 en conservant les maximums par fenêtres de deux valeurs. `AdaptiveAvgPool1d(4)` ramène ensuite chaque filtre à quatre valeurs moyennes.

`Flatten` rassemble ces **32 × 4 = 128 valeurs** dans un vecteur. `Linear` les transforme en quatre scores, un par classe. Ces scores ne sont pas directement des probabilités.

### Entraîner les poids

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

Adam ajuste les poids du réseau avec un taux d’apprentissage de `0.001`. `CrossEntropyLoss` mesure l’erreur entre les scores produits et les classes attendues.

À chaque lot, la boucle d’entraînement effectue ces opérations :

```python
optimizer.zero_grad()
logits = model(batch_x)
loss = criterion(logits, batch_y)
loss.backward()
optimizer.step()
```

`zero_grad` efface les gradients du lot précédent. `model(batch_x)` calcule les scores et `criterion` calcule l’erreur. `backward` détermine comment chaque poids contribue à cette erreur, puis `step` met les poids à jour. Le programme parcourt ainsi les données **30 fois**, soit 30 époques.

### Suivre et évaluer l’apprentissage

```python
history["loss"].append(total_loss / len(train_y))
history["accuracy"].append(correct / len(train_y))
```

Ces listes enregistrent la perte moyenne et la proportion de bonnes réponses pendant chaque époque d’entraînement. La cellule suivante trace leur progression.

```python
model.eval()
with torch.no_grad():
    predictions = model(test_x).argmax(dim=1)
    accuracy = (predictions == test_y).float().mean().item()
```

`eval` active le mode évaluation et `no_grad` évite le calcul des gradients. `argmax` retient le numéro de la classe ayant le score le plus élevé. La comparaison avec `test_y` donne la proportion de bonnes prédictions sur le test.

`ConfusionMatrixDisplay.from_predictions(...)` détaille ces résultats : les lignes représentent les classes réelles et les colonnes les classes prédites. La diagonale contient les bonnes réponses.

### Transférer et réduire le modèle

```python
weights = [layer.weight.detach().cpu().numpy() for layer in (model[0], model[3], model[7])]
biases = [layer.bias.detach().cpu().numpy() for layer in (model[0], model[3], model[7])]
```

Ces lignes récupèrent les poids et les biais des deux convolutions et de la couche finale. `detach` les sépare du calcul des gradients, `cpu` les place en mémoire CPU et `numpy` les rend utilisables par TensorFlow.

La classe `TensorFlowCNN` reproduit les opérations du réseau avec ces paramètres. Les transpositions adaptent l’ordre des dimensions entre PyTorch et TensorFlow. **Cette étape transfère le modèle déjà entraîné ; elle ne lance aucun nouvel apprentissage.**

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
int8_model = converter.convert()
```

Le convertisseur produit une version TensorFlow Lite entièrement quantifiée en **INT8**, avec des valeurs représentées sur 8 bits. `representative_data` fournit les séquences d’entraînement pour calibrer cette conversion. Les données de test ne servent pas à cette calibration.

La cellule sauvegarde une référence Float32 et une version INT8 dans `models/`. La dernière cellule transforme les octets de `cnn1d_int8.tflite` en un tableau C++ dans `src/gesture_model.cpp`. Elle génère aussi `include/gesture_model.h`, qui expose le modèle, les classes, les dimensions et les paramètres de quantification au firmware.

## Le programme main.cpp

### Déclarer l’écran, le capteur et la mémoire du modèle

```cpp
LiquidCrystal lcd(1, 2, 3, 4, 5, 8);
constexpr int kSda = 7, kScl = 6;
constexpr uint32_t kSamplePeriodMs = 20;
alignas(16) uint8_t tensorArena[64 * 1024];
```

Les six broches de `LiquidCrystal` correspondent à **RS, E, D4, D5, D6 et D7**. Le MPU6050 utilise SDA sur GPIO 7 et SCL sur GPIO 6.

Une période de 20 ms donne **50 mesures par seconde**. `tensorArena` réserve 64 Ko de RAM pour les données intermédiaires de TensorFlow Lite Micro ; `alignas(16)` aligne cette mémoire sur 16 octets.

`show` efface l’écran et écrit ses deux lignes. `stop` affiche une erreur sur l’écran et le port série, puis bloque le programme jusqu’au redémarrage.

### Lire et réorienter les mesures

`writeRegister` configure un registre du MPU6050 via I²C. `readRegisters` lit plusieurs octets à partir d’un registre et vérifie que le capteur répond.

Dans `readMotion`, une lecture de 14 octets à partir de `0x3B` récupère les accélérations, la température et les rotations. Le code sélectionne les octets utiles :

```cpp
const int offsets[] = {0, 2, 12, 8};
int16_t raw = static_cast<int16_t>((data[i] << 8) | data[i + 1]);
values[axis] = raw / (axis < 2 ? 16384.0f : 131.0f);
```

`offsets` indique où commence chaque valeur dans les 14 octets. Le décalage `<< 8` et le `|` réunissent les deux octets d’une mesure en un entier signé de 16 bits. La division exprime les accélérations en **g** et les rotations en **degrés par seconde**, pour les plages configurées.

La permutation demandée est appliquée directement lors de cette sélection :

| Entrée du modèle | Mesure physique | Position dans les octets |
| --- | --- | --- |
| X | Accélération X | 0 |
| Y | Accélération Y | 2 |
| GX | Gyroscope Z | 12 |
| GY | Gyroscope X | 8 |

Le gyroscope Y devient le nouveau GZ, que le modèle n’utilise pas.

### Initialiser le matériel et charger le modèle

`setup()` initialise l’écran en 16 × 2 caractères et le bus I²C. Il recherche le MPU6050 aux adresses `0x68` puis `0x69`, le réveille et règle ses plages sur **±2 g** et **±250 °/s**. Les registres du filtre et du diviseur fixent sa cadence à 50 Hz.

```cpp
const tflite::Model* model = tflite::GetModel(g_gesture_model);
static tflite::MicroInterpreter runner(model, resolver, tensorArena, sizeof(tensorArena));
```

`GetModel` lit le modèle contenu dans le tableau C++. Les appels `resolver.Add...` déclarent les huit types d’opérations nécessaires à son exécution. La convolution 1D est représentée par une opération `Conv2D` dans l’export TFLite.

`MicroInterpreter` associe le modèle, ses opérations et la mémoire de calcul. `AllocateTensors()` prépare les tenseurs. `input` et `output` donnent ensuite accès aux zones d’entrée et de sortie. Le code vérifie les types INT8 et les dimensions attendues avant de commencer la capture.

### Faire le décompte et préparer les entrées

Dans `loop()`, une boucle affiche successivement 3, 2 et 1, avec `delay(1000)` entre chaque nombre. Après le décompte, le programme lit 75 mesures et traite leurs quatre axes :

```cpp
float normalized = (values[axis] - kMinimum[axis]) / (kMaximum[axis] - kMinimum[axis]);
normalized = constrain(normalized, 0.0f, 1.0f);
int quantized = lroundf(normalized / input->params.scale) + input->params.zero_point;
input->data.int8[sample * 4 + axis] = constrain(quantized, -128, 127);
```

La première ligne applique la normalisation min-max du dataset. `constrain` limite le résultat à `[0, 1]`. La valeur est ensuite convertie en entier INT8 à l’aide de l’échelle et du décalage lus dans le modèle.

`sample * 4 + axis` range les valeurs dans l’ordre **X, Y, GX, GY**, mesure après mesure. `nextSample` avance de 20 ms à chaque capture et le programme attend cette échéance avant de poursuivre : les 75 mesures occupent ainsi 1,5 seconde.

Les tableaux `kMinimum` et `kMaximum` reprennent les bornes de [normalisation.json](dataset_thomas_normalised/normalisation.json). Si le dataset change, il faut actualiser ces constantes manuellement en plus de réexporter le modèle.

### Exécuter le modèle et afficher le signe

```cpp
if (interpreter->Invoke() != kTfLiteOk) stop("Reconnaissance");
int predicted = 0;
for (int i = 1; i < kGestureClasses; ++i)
  if (output->data.int8[i] > output->data.int8[predicted]) predicted = i;
show("Signe detecte :", kGestureLabels[predicted]);
```

`Invoke` lance la reconnaissance sur les mesures capturées. La boucle retient l’indice du plus grand score, puis `kGestureLabels` fournit le nom de la classe à afficher. Ce résultat est aussi envoyé sur le port série à 115200 bauds.

Le programme affiche le résultat pendant trois secondes, puis recommence le décompte. Il choisit toujours l’une des quatre classes ; aucun seuil de confiance supplémentaire n’est appliqué.

## Exécuter le projet

Pour ouvrir le notebook depuis la racine :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab src/model_train.ipynb
```

Sélectionner le noyau de cet environnement et exécuter toutes les cellules pour entraîner et exporter le modèle. Depuis le terminal PlatformIO, compiler, téléverser puis ouvrir le moniteur série :

```bash
pio run
pio run --target upload
pio device monitor
```

Pour l’écran, RW, VSS et K sont reliés à GND, VDD et A au 5 V, et V0 règle le contraste. Les masses des périphériques et de l’ESP32 sont communes.

Le firmware a été compilé pour l’ESP32-S3 ; son fonctionnement sur le montage physique reste à vérifier.
