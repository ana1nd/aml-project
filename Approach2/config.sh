'''
config file for training encoders 
'''

python train_nli.py --nlipath dataset/MNLI \
                    --datatype mnli \
                    --outputmodelname LSTM.M.100.pickle \
                    --encodermodelname LSTM.M.100.pickle \
                    --n_epochs 20 \
                    --encoder_type LSTMEncoder \
                    --enc_lstm_dim 100 \
                    --n_enc_layers 1 \
                    --fc_dim 512 \
                    --pool_type max \
                    --gpu_id 1 \
                    --batch_size 256



                    #--dpout_model 0.1 \
                    #--optimizer adam,lr=0.1
                    #--optimizer "adam,lr=0.1,beta1=0.9,beta2=0.999,weight_decay=0,eps=0.0000001" \