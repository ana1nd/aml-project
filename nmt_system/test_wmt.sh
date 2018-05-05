# script for applying saved model once model is trained on SNLI data.

# bash test_wmt.sh $1 $2 --encoder_hidden_dim $3 --refpath $files --hyppath $files3 --lp $i

model_name=$1
dim=$2
#file_name=$3

mkdir -p logs/$model_name

# remove files if already exists
# sys_file=logs/$model_name/$model_name".sys.score"
# if [ -f  $sys_file ] ; then
#     rm $sys_file
# fi

# seg_file=logs/$model_name/$model_name".seg.score"
# if [ -f  $seg_file ] ; then
#     rm $seg_file
# fi


## now loop through the above array

python score_sts_es.py --outputmodelname $1 --enc_lstm_dim $2 --flag "first" --inputfile test_data_es/sts15.input.newswire.txt --gsfile test_data_es/sts15.gs.newswire.txt
python score_sts_es.py --outputmodelname $1 --enc_lstm_dim $2 --flag "second" --inputfile test_data_es/sts15.input.wikipedia.txt --gsfile test_data_es/sts15.gs.wikipedia.txt
python score_sts_es.py --outputmodelname $1 --enc_lstm_dim $2 --flag "third" --inputfile test_data_es/sts17.input.track3.es-es.txt --gsfile test_data_es/sts17.gs.track3.es-es.txt

mv logs/*.seg.score logs/$model_name/
