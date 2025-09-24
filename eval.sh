conda init
conda activate HEIST

model_name="HEIST"
echo "$model_name"
python calculate_rep.py --data_name dfci --model_name $model_name
python calculate_rep.py --data_name upmc --model_name $model_name
python calculate_rep.py --data_name charville --model_name $model_name
python calculate_rep.py --data_name sea --model_name $model_name
python calculate_rep.py --data_name melanoma --model_name $model_name
python calculate_rep.py --data_name placenta --model_name $model_name
python calculate_rep.py --data_name lung --model_name $model_name

python eval_tissue_classification.py  --data_name dfci --label_name pTR_label --model_name $model_name

python eval_tissue_classification.py  --data_name charville --label_name primary_outcome --model_name $model_name
python eval_tissue_classification.py  --data_name charville --label_name recurrence --model_name $model_name

python eval_tissue_classification.py  --data_name upmc --label_name primary_outcome --model_name $model_name
python eval_tissue_classification.py  --data_name upmc --label_name recurred --model_name $model_name

python eval_melanoma.py --model_name $model_name
python eval_cell_clustering.py $model_name
python eval_placenta.py --model_name $model_name --num_layers 4

python eval_gene_inputation.py --model_name $model_name --data_name melanoma
python eval_gene_inputation.py --model_name $model_name --data_name placenta
python eval_gene_inputation_fine_tune.py --model_name $model_name --data_name melanoma
python eval_gene_inputation_fine_tune.py --model_name $model_name --data_name placenta