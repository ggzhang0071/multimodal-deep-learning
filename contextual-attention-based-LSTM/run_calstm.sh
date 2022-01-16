#python create_data.py

mkdir Logs
#python run.py --unimodal True --fusion True --data "iemocap"   --classes 6  2>&1 |tee Logs/run_unimodal.log
python run.py --unimodal False --fusion True --data "iemocap" --batch_size 256  --classes 6  2>&1 |tee Logs/run_multimodal.log

