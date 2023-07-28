
"""
This file converts a csv to the format needed for tabddpm
"""
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def convert_csv_to_data(real_data_dir, real_data_path, target, num_cols=[], task_type='binclass'):
    os.chdir(real_data_dir)
    
    real = pd.read_csv(real_data_path)
    
    if len(num_cols) < 1:
    	num_cols = real.select_dtypes(include='number').columns.tolist()
    	num_cols = [c for c in num_cols if c != target] # Ensure target excluded
    print(real_data_path, num_cols)
   
    cat_cols = [c for c in real.columns if (c not in num_cols and c != target)]
    if task_type != 'regression':
        # Task 1: Create a dictionary representing a mapping from each category to its index
        category_mapping = {category: index for index, category in enumerate(real[target].unique())}
        inverse_mapping = {index: category for index, category in enumerate(real[target].unique())}
        inverse_mapping_str = {str(key): str(value) for key, value in inverse_mapping.items()}
        #print(inverse_mapping)
        #print([(type(index), type(category)) for index, category in enumerate(real[target].unique())])
        # Task 2: Map all values in the column based on the mapping
        real[target] = real[target].map(category_mapping)
    else:
    	inverse_mapping_str = None
    target_codes = real[target].values
    

    # Generate a list of indices from 0 to len(real)-1
    indices = np.arange(len(real))
    # First split to separate out the training set (80%)
    train_idx, rest_idx = train_test_split(indices, test_size=0.2, random_state=42)
    # Then split the remaining data to separate out the validation (15%) and test (5%) sets
    test_idx, val_idx = train_test_split(rest_idx, test_size=0.75, random_state=42)
    # Now use these indices to create the training, validation and test sets
    idx_by_splits = {'train':train_idx, 'test':test_idx, 'val':val_idx}

    for split,idx in idx_by_splits.items():
        data_this_split = real.iloc[idx]
        X_cat = data_this_split[cat_cols].to_numpy().astype('<U26')
        X_num = data_this_split[num_cols].to_numpy()
        data_this_split[target] = pd.Categorical(data_this_split[target])
        y = target_codes[idx]
        np.save(f"X_num_{split}.npy", X_num)
        np.save(f"X_cat_{split}.npy", X_cat)
        np.save(f"y_{split}.npy", y)
        np.save(f"idx_{split}.npy", idx)
        
    exp_name = real_data_path.split('/')[-1]
    n_classes = 0 if task_type == 'regression' else len(real[target].unique())
    info_json = {
        "name": exp_name,
        "id": f"{exp_name}--default",
        "task_type": task_type,
        "n_num_features": len(num_cols),
        "n_cat_features": len(cat_cols),
        "test_size": len(test_idx),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "n_classes":n_classes
    }
    with open('info.json', 'w') as json_file:
        json_file.write(json.dumps(info_json))
    column_names_json = {'cat':cat_cols, 'num':num_cols, 'target':target, 'test_name':exp_name, 'inverse_mapping':inverse_mapping_str}
    with open('col_names.json', 'w') as json_file:
        json_file.write(json.dumps(column_names_json, indent=4))
        
        
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/adult', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/adult/adult.csv', 
                    target='income', 
                    task_type='binclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/absent', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/absent/absent.csv', 
                    num_cols = ['Month of absence', 'Day of the week',
                               'Transportation expense', 'Distance from Residence to Work',
                               'Service time', 'Age', 'Work load Average/day ', 'Hit target',
                               'Disciplinary failure', 'Education','Pet', 'Weight', 'Height', 'Body mass index'],
                    target='Absenteeism time in hours',task_type='regression')  
                  
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/abalone', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/abalone/abalone.csv', 
                    target='Rings',task_type='regression')  
                  
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Churn_Modelling', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Churn_Modelling/Churn_Modelling.csv', 
                    target='Exited',task_type='binclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Bean', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Bean/Bean.csv', 
                    target='Class',task_type='multiclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Beijing', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Beijing/Beijing.csv', 
                    target='pm2.5',task_type='regression')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/faults', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/faults/faults.csv', 
                    num_cols = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
                               'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
                               'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
                               'Steel_Plate_Thickness',
                               'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
                               'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
                               'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
                               'SigmoidOfAreas'],
                    target='Other_Faults',task_type='multiclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/HTRU_2', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/HTRU_2/HTRU_2.csv', 
                    target='Class',task_type='binclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/insurance', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/insurance/insurance.csv', 
                    target='charges',task_type='regression')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Magic', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Magic/Magic.csv', 
                    target='class',task_type='binclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/News', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/News/News.csv', 
                    target=' shares',task_type='regression')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/nursery', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/nursery/nursery.csv', 
                    target='final evaluation',task_type='multiclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Obesity', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Obesity/Obesity.csv', 
                    target='NObeyesdad',task_type='multiclass')  
               
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Titanic', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/Titanic/Titanic.csv', 
                    target='Embarked',task_type='multiclass')  
                    
convert_csv_to_data(real_data_dir='/media/xflin/10tb/Xiaofeng/TabDDPM/data/wilt', 
                    real_data_path='/media/xflin/10tb/Xiaofeng/TabDDPM/data/wilt/wilt.csv', 
                    target='class',task_type='binclass')  
                    
                   
                    
