cd code

python train_local_models.py 2p_A
python train_local_models.py 2p_B
python train_local_models.py 2p_C
python train_local_models.py 3p_A
python train_local_models.py 3p_B
python train_local_models.py 7p

FOR /L %%random_seed IN (1,1,10) DO (
  python HMR.py 2p_A %%random_seed
  python HMR.py 2p_B %%random_seed
  python HMR.py 2p_C %%random_seed
  python HMR.py 3p_A %%random_seed
  python HMR.py 3p_B %%random_seed
  python HMR.py 7p %%random_seed
)

python plot_exp_results.py