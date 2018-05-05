'''
config file for training encoders 
'''

python train_nli.py --nlipath dataset/SNLI \
                    --datatype snli \
                    --outputmodelname LSTM.2048.pickle \
                    --encodermodelname LSTM.2048.pickle \
                    --n_epochs 25 \
                    --encoder_type LSTMEncoder \
                    --enc_lstm_dim 2048 \
                    --dpout_model 0.2 \
                    --dpout_fc 0.2 \
                    --n_enc_layers 1 \
                    --fc_dim 512 \
                    --pool_type max \
                    --gpu_id 3 \
                    --batch_size 128 



                    #--dpout_model 0.1 \
                    #--optimizer adam,lr=0.1
                    #--optimizer "adam,lr=0.1,beta1=0.9,beta2=0.999,weight_decay=0,eps=0.0000001" \