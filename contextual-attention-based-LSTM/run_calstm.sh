#python create_data.py

mkdir Logs
python run.py --unimodal True --fusion True --data "iemocap"  --batch_size 1024  --classes 6 2&>1 |tee Logs/run.log
#python run.py --unimodal False --fusion True

