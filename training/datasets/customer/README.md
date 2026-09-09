# Customer dataset layout (Faza 3)
#
# training/datasets/customer/
# ├── normal/          # 500–5000 slik dobrih kosov
# │   └── *.jpg
# ├── defect/          # 50–500 slik z napakami (opcijsko označene)
# │   └── *.jpg
# └── metadata.csv     # part_number,product_type,batch,timestamp,machine,defect_type,qc_result
#
# Trening anomaly:
#   python training/train_anomaly.py --dataset customer --output weights/patchcore_customer
#
# YOLO fine-tune: pripravi YOLO labels in zaženi train_yolo_gc10.py z njihovim data.yaml.
