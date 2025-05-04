DATA_DIR = "data"
ANALYSIS_PATH = "results"

# CIC-IDS2017
EXTENDED_CIC_IDS2017_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip"
CIC_IDS2017_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip"

CIC_IDS2017_CLASSES_MAPPING = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Bot',
    'Web Attack  Brute Force': 'Web Attack',
    'Web Attack  Sql Injection': 'Web Attack',
    'Web Attack  XSS': 'Web Attack',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed'
}

CICIDS2017_DEFAULT_CONFIG = {
    "pca": True,
    "classes_mapping": True
}

# UTILS
SMOTE_CONFIG = {
    "class_samples_threshold": 0.3,
    "n": 10,
    "alpha":  0.7,
    "beta": 0.3
}