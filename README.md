# PCB Defect Detection YOLOv5

Application Python pour detecter les defauts de cartes PCB avec un modele YOLOv5 entraine.

Le script ouvre la camera, detecte les defauts en temps reel, stabilise les detections sur plusieurs frames, puis exporte une inspection avec :

- une image annotee ;
- un fichier JSON ;
- un rapport HTML ;
- les fichiers `PCB_latest.json` et `PCB_latest.jpg` pour une integration Unity ou une autre application.

## Contenu du depot

- `CAM.py` : script principal de detection.
- `best.pt` : poids du modele YOLOv5 entraine.
- `requirements.txt` : dependances Python necessaires.
- `.gitignore` : ignore les fichiers locaux, les environnements virtuels, les datasets et les sorties generees.

Les dossiers `inspections/`, `captures_pcb/`, `images/`, `.venv/`, `yolov5/` et `XmlToTxt/` ne sont pas versionnes car ce sont des donnees locales, des sorties generees, ou des dependances externes.

## Installation

```bash
git clone https://github.com/merysmk/pcb-defect-detection-yolov5.git
cd pcb-defect-detection-yolov5
python -m venv .venv
```

Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS :

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Lancer la detection :

```bash
python CAM.py
```

Appuyer sur `ESC` pour terminer une inspection et exporter les resultats.

Par defaut, le script cherche le modele dans `best.pt`. Pour utiliser un autre fichier de poids :

Windows PowerShell :

```powershell
$env:WEIGHTS_PATH="chemin\vers\modele.pt"
python CAM.py
```

Linux/macOS :

```bash
WEIGHTS_PATH="chemin/vers/modele.pt" python CAM.py
```

## Sorties generees

Les resultats sont crees dans le dossier `inspections/` :

- `inspection_DATE/defects.json`
- `inspection_DATE/report.html`
- `inspection_DATE/frame_final.jpg`
- `PCB_latest.json`
- `PCB_latest.jpg`

Ces fichiers ne sont pas pousses sur GitHub pour garder le depot propre.

## Notes

- Une camera doit etre connectee et accessible.
- Au premier lancement, `torch.hub` peut telecharger YOLOv5 depuis GitHub.
- Si l'installation de PyTorch pose probleme, installer la version adaptee a votre machine depuis le site officiel de PyTorch, puis relancer `pip install -r requirements.txt`.
