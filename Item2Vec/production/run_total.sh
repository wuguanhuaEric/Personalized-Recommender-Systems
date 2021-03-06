python="\Programs\Python\Python37"
user_rating_file="../data/ratings.txt"
train_file="../data/train_data_bk.txt"
item_vec_file="../data/item_vec_bk.txt"
item_sim_file="../data/sim_result_bk.txt"

if [-f user_rating_file];then
    $python produce_train_data.py $user_rating_file $train_file
else
    echo "no rating file"
    exit
fi
if [-f $train_file]; then
    sh train.sh $train_file $item_vec_file
else
    echo "no train file"
    exit
fi
if [-f $item_vec_file];then
    $python produce_item_sim.py $item_vec_file $item_sim_file
else
    echo "no item vec file"
    exit
fi