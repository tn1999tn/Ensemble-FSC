# training PN model
# python write-config/write_yaml_PN.py
# python individual.py --cfg configs/PN/miniImageNet_res12.yaml --tag main

# searching for hyperparameters for finetune.
# python write-config/write_yaml_search.py
# python search_hyperparameter.py --cfg configs/search/ft_res12_PN.yaml

#test individual
#python write-config/write_yaml_test.py
#python individual.py --cfg configs/mini_evaluation_5/ft_res12_pn.yaml --tag test

#test ensemble
#python test-ensemble.py --cfg configs/mini_evaluation_5/cc_res12_ce.yaml --cfg1 configs/mini_evaluation_5/cc_res12_feat.yaml --cfg2 cc_res12_ier.yaml --cfg3 cc_res12_pn.yaml --cfg4 frn_res12_cc.yaml --cfg5 frn_res12_ft.yaml --cfg6 frn_res12_tsa.yaml
# --cfg7 ft_res12_ce.yaml --cfg8 ft_res12_feat.yaml --cfg9 ft_res12_ier.yaml --cfg10 ft_res12_pn.yaml --cfg11 pool_res12_cc.yaml --cfg12 pool_res12_ft.yaml --cfg13 pool_res12_tsa.yaml --cfg14 tsa_res12_ce.yaml --cfg15 tsa_res12_feat.yaml --cfg16 tsa_res12_ier.yaml --cfg17 tsa_res12_pn.yaml --tag test



