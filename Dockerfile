#Version Python
FROM python:3.12.3

# Créer le dossier de travail
WORKDIR /app

# Copier les fichiers du projet
COPY marketing_campaign.csv /app/
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Lancer train.py puis les tests
CMD ["bash", "-c", "python train.py && pytest test_preprocess.py"]