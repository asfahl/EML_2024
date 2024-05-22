# conda activate /mnt/hd1/conda/aimet

# 10.1.1
# compute_encodings
# Berechnet die optimale Quantisierung der Gewicht und Aktivierungsfkt. eines trainierten
# Models. Ein representativer Datensatz ist zusätzlich erforderlich.
#
# forward_pass_callback
# Der Parameter von compute_encodings ist eine Callback-Fkt. die die Forward-Fkt. des Modells auf einem gegebenen Datansatz aufruft.
# Auf Basis des durchlaufs der Forward Funktion errechnet compute_emcodings dann die Quantisierung.  